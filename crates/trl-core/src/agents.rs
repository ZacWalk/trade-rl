//! Agents: constant positions, rules, a Q-table, a double DQN, and the
//! ensemble vote across seeds.

use crate::env::{ACTION_COUNT, action_name, action_to_position, position_to_action};
use crate::error::{Result, invalid};
use crate::features::FeatureMatrix;
use crate::math::clamp;
use crate::mlp::{Mlp, network_seed, to_f32};
use crate::rng::Rng;
use crate::text::general;

/// Every agent `trl walk --agent` accepts.
pub const AGENT_NAMES: [&str; 8] = [
    "flat",
    "long",
    "short",
    "random",
    "crossover",
    "breakout",
    "qtable",
    "dqn",
];

/// Picks an action (0 short, 1 flat, 2 long) from an environment state:
/// features, then position, unrealised pnl and holding time.
pub trait Agent: Send {
    fn name(&self) -> &str;
    fn act(&mut self, state: &[f64], greedy: bool) -> Result<usize>;
    fn observe(&mut self, _state: &[f64], _action: usize, _reward: f64, _next: &[f64], _done: bool) -> Result<()> {
        Ok(())
    }
    fn begin_episode(&mut self) {}
    fn end_epoch(&mut self) -> Result<()> {
        Ok(())
    }
    fn learns(&self) -> bool {
        false
    }
    fn exploration(&self) -> f64 {
        0.0
    }
    fn recent_loss(&self) -> f64 {
        0.0
    }
    /// The learned policy as an artifact: file name and bytes.
    fn policy(&self) -> Option<(&'static str, Vec<u8>)> {
        None
    }
}

/// The position held, from the state's position column.
fn held(state: &[f64]) -> i32 {
    state[state.len() - 3] as i32
}

/// Always takes the same side: the bar every learner has to clear.
pub struct Fixed {
    action: usize,
    label: &'static str,
}

impl Agent for Fixed {
    fn name(&self) -> &str {
        self.label
    }

    fn act(&mut self, _: &[f64], _: bool) -> Result<usize> {
        Ok(self.action)
    }
}

/// Trades at a chosen rate without an opinion: the real cost floor to beat.
pub struct RandomAgent {
    flip: f64,
    rng: Rng,
    held: usize,
}

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        "random"
    }

    fn act(&mut self, _: &[f64], _: bool) -> Result<usize> {
        if self.rng.uniform() < self.flip {
            self.held = self.rng.index(ACTION_COUNT);
        }
        Ok(self.held)
    }

    fn begin_episode(&mut self) {
        self.held = 1;
    }
}

/// Trades the fast/slow trend spread with a dead band.
pub struct Crossover {
    threshold: f64,
    trend_column: usize,
}

impl Agent for Crossover {
    fn name(&self) -> &str {
        "crossover"
    }

    fn act(&mut self, state: &[f64], _: bool) -> Result<usize> {
        let trend = state[self.trend_column];
        Ok(if trend > self.threshold {
            position_to_action(1)
        } else if trend < -self.threshold {
            position_to_action(-1)
        } else {
            // Inside the dead band, keep whatever is held.
            position_to_action(held(state))
        })
    }
}

/// The hand-coded form of the one effect that survived an earlier pattern
/// study: drift is positive above the previous session's high. It exists so
/// a learner has to beat it, not because a fixed rule is the goal.
pub struct Breakout {
    threshold: f64,
    high_column: usize,
}

impl Agent for Breakout {
    fn name(&self) -> &str {
        "breakout"
    }

    fn act(&mut self, state: &[f64], _: bool) -> Result<usize> {
        // Distance beyond the previous session's high, in previous-day ranges.
        let beyond = state[self.high_column];
        Ok(if beyond > self.threshold || (held(state) > 0 && beyond > 0.0) {
            position_to_action(1)
        } else {
            position_to_action(0)
        })
    }
}

fn decayed(start: f64, end: f64, step: i64, over: i64) -> f64 {
    if over <= 0 {
        return end;
    }
    start + (end - start) * clamp(step as f64 / over as f64, 0.0, 1.0)
}

fn require(valid: bool, learner: &str, rule: &str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(invalid(format!("invalid {learner} configuration: {rule}")))
    }
}

/// Epsilon is a probability; a decay over zero steps starts at the end rate.
fn valid_exploration(start: f64, end: f64, over: i64) -> bool {
    (0.0..=1.0).contains(&start) && (0.0..=1.0).contains(&end) && over >= 0
}

#[derive(Clone, Debug, PartialEq)]
pub struct TabularConfig {
    pub learning_rate: f64,
    pub discount: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay_steps: i64,
    pub reward_scale: f64,
}

impl Default for TabularConfig {
    fn default() -> TabularConfig {
        TabularConfig {
            learning_rate: 0.1,
            discount: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 200_000,
            reward_scale: 10_000.0,
        }
    }
}

const TREND_EDGES: [f64; 4] = [-1.0, -0.3, 0.3, 1.0];
const RETURN_EDGES: [f64; 4] = [-1.0, -0.3, 0.3, 1.0];
const VOL_EDGES: [f64; 2] = [-0.3, 0.3];

fn bucket(value: f64, edges: &[f64]) -> usize {
    edges.iter().take_while(|&&e| value >= e).count()
}

/// Small enough that the whole learned policy can be printed and read.
pub struct TabularQ {
    config: TabularConfig,
    feature_columns: usize,
    trend_column: usize,
    return_column: usize,
    volatility_column: usize,
    q: Vec<f64>,
    visits: Vec<i32>,
    rng: Rng,
    steps: i64,
    loss: f64,
}

impl TabularQ {
    pub fn new(config: &TabularConfig, features: &FeatureMatrix, seed: u64) -> Result<TabularQ> {
        // A rate outside (0, 1] or a discount above 1 diverges or freezes the
        // table instead of training it.
        require(
            config.learning_rate > 0.0 && config.learning_rate <= 1.0,
            "Q-table",
            "--q-lr must be in (0, 1]",
        )?;
        require(
            (0.0..=1.0).contains(&config.discount),
            "Q-table",
            "--gamma must be in [0, 1]",
        )?;
        require(
            valid_exploration(config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps),
            "Q-table",
            "--epsilon-decay must be at least 0",
        )?;
        require(
            config.reward_scale > 0.0 && config.reward_scale.is_finite(),
            "Q-table",
            "the reward scale must be positive",
        )?;
        let states = 5 * 5 * 3 * 3;
        Ok(TabularQ {
            config: config.clone(),
            feature_columns: features.columns,
            trend_column: features.column("trend_fast")?,
            return_column: features.column("ret_60")?,
            volatility_column: features.column("vol_ratio")?,
            q: vec![0.0; states * ACTION_COUNT],
            visits: vec![0; states],
            rng: Rng::new(seed),
            steps: 0,
            loss: 0.0,
        })
    }

    fn discretise(&self, state: &[f64]) -> usize {
        let trend = bucket(state[self.trend_column], &TREND_EDGES);
        let ret = bucket(state[self.return_column], &RETURN_EDGES);
        let vol = bucket(state[self.volatility_column], &VOL_EDGES);
        let position = (state[self.feature_columns] as i32 + 1).clamp(0, 2) as usize;
        ((trend * 5 + ret) * 3 + vol) * 3 + position
    }

    fn best_action(&self, index: usize) -> usize {
        let q = &self.q[index * ACTION_COUNT..(index + 1) * ACTION_COUNT];
        let mut best = 0;
        for a in 1..ACTION_COUNT {
            if q[a] > q[best] {
                best = a;
            }
        }
        best
    }

    pub fn visited_states(&self) -> usize {
        self.visits.iter().filter(|&&v| v > 0).count()
    }

    pub fn state_count(&self) -> usize {
        self.visits.len()
    }
}

impl Agent for TabularQ {
    fn name(&self) -> &str {
        "qtable"
    }

    fn act(&mut self, state: &[f64], greedy: bool) -> Result<usize> {
        let index = self.discretise(state);
        if !greedy && self.rng.uniform() < self.exploration() {
            return Ok(self.rng.index(ACTION_COUNT));
        }
        Ok(self.best_action(index))
    }

    fn observe(&mut self, state: &[f64], action: usize, reward: f64, next: &[f64], done: bool) -> Result<()> {
        let index = self.discretise(state);
        let next_index = self.discretise(next);
        self.visits[index] += 1;
        self.steps += 1;
        let bootstrap = if done {
            0.0
        } else {
            self.config.discount * self.q[next_index * ACTION_COUNT + self.best_action(next_index)]
        };
        let target = reward * self.config.reward_scale + bootstrap;
        let value = &mut self.q[index * ACTION_COUNT + action];
        let error = target - *value;
        *value += self.config.learning_rate * error;
        self.loss += (error * error - self.loss) * 0.001;
        Ok(())
    }

    fn learns(&self) -> bool {
        true
    }

    fn exploration(&self) -> f64 {
        decayed(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.steps,
            self.config.epsilon_decay_steps,
        )
    }

    fn recent_loss(&self) -> f64 {
        self.loss
    }

    fn policy(&self) -> Option<(&'static str, Vec<u8>)> {
        let mut out = String::from("state,visits,q_short,q_flat,q_long,policy\n");
        for (s, visits) in self.visits.iter().enumerate() {
            let q = &self.q[s * ACTION_COUNT..(s + 1) * ACTION_COUNT];
            out += &format!(
                "{s},{visits},{},{},{},{}\n",
                general(q[0], 6),
                general(q[1], 6),
                general(q[2], 6),
                action_name(self.best_action(s))
            );
        }
        Some(("policy.csv", out.into_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DqnConfig {
    pub hidden: Vec<usize>,
    pub learning_rate: f64,
    /// Applied after each training epoch; 1 is off.
    pub lr_decay: f64,
    pub discount: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay_steps: i64,
    /// Rewards become basis points, so Q is O(1).
    pub reward_scale: f64,
    pub replay_capacity: usize,
    pub replay_warmup: usize,
    pub batch_size: usize,
    pub train_every: usize,
    pub target_sync: usize,
    pub huber_delta: f64,
    /// Q error is larger than the cost of switching, so argmax flips on
    /// noise. Requiring the edge to cover the move is the right decision
    /// rule; 0 is off.
    pub switch_margin: f64,
}

impl Default for DqnConfig {
    fn default() -> DqnConfig {
        DqnConfig {
            hidden: vec![64, 64],
            learning_rate: 3e-4,
            lr_decay: 1.0,
            discount: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 200_000,
            reward_scale: 10_000.0,
            replay_capacity: 200_000,
            replay_warmup: 5000,
            batch_size: 32,
            train_every: 4,
            target_sync: 2000,
            huber_delta: 1.0,
            switch_margin: 0.0,
        }
    }
}

/// Double DQN with experience replay and a target network.
pub struct Dqn {
    config: DqnConfig,
    state_size: usize,
    online: Mlp,
    target: Mlp,
    rng: Rng,
    states: Vec<f32>,
    next_states: Vec<f32>,
    actions: Vec<u8>,
    rewards: Vec<f64>,
    dones: Vec<bool>,
    stored: usize,
    cursor: usize,
    batch_states: Vec<f32>,
    batch_next: Vec<f32>,
    batch_grad: Vec<f32>,
    batch_target: Vec<f64>,
    batch_index: Vec<usize>,
    input: Vec<f32>,
    steps: i64,
    loss: f64,
}

impl Dqn {
    pub fn new(config: &DqnConfig, state_size: usize, seed: u64) -> Result<Dqn> {
        require(config.batch_size > 0, "DQN", "--batch must be at least 1")?;
        require(config.target_sync > 0, "DQN", "--target-sync must be at least 1")?;
        require(
            config.learning_rate.is_finite() && config.learning_rate > 0.0,
            "DQN",
            "--lr must be positive",
        )?;
        require(
            config.lr_decay.is_finite() && config.lr_decay > 0.0,
            "DQN",
            "--lr-decay must be positive",
        )?;
        require(
            (0.0..=1.0).contains(&config.discount),
            "DQN",
            "--gamma must be in [0, 1]",
        )?;
        require(
            valid_exploration(config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps),
            "DQN",
            "--epsilon-decay must be at least 0",
        )?;
        require(config.switch_margin >= 0.0, "DQN", "--switch-margin must be at least 0")?;
        require(
            state_size > 0
                && config.replay_capacity > 0
                && config.replay_warmup > 0
                && config.replay_warmup <= config.replay_capacity
                && config.train_every > 0
                && !config.hidden.is_empty()
                && config.hidden.iter().all(|&h| h > 0 && h <= 1 << 20),
            "DQN",
            "bad replay or layer settings",
        )?;
        let mut sizes = vec![state_size];
        sizes.extend(&config.hidden);
        sizes.push(ACTION_COUNT);
        let mut online = Mlp::new(&sizes, network_seed(seed))?;
        let mut target = Mlp::new(&sizes, network_seed(seed))?;
        target.copy_from(&online)?;
        online.set_learning_rate(to_f32(config.learning_rate)?)?;
        let capacity = config.replay_capacity;
        let batch = config.batch_size;
        Ok(Dqn {
            config: config.clone(),
            state_size,
            online,
            target,
            rng: Rng::new(seed),
            states: vec![0.0; capacity * state_size],
            next_states: vec![0.0; capacity * state_size],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            dones: vec![false; capacity],
            stored: 0,
            cursor: 0,
            batch_states: vec![0.0; batch * state_size],
            batch_next: vec![0.0; batch * state_size],
            batch_grad: vec![0.0; batch * ACTION_COUNT],
            batch_target: vec![0.0; batch],
            batch_index: vec![0; batch],
            input: vec![0.0; state_size],
            steps: 0,
            loss: 0.0,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.online.parameter_count()
    }

    pub fn learning_rate(&self) -> f64 {
        f64::from(self.online.learning_rate())
    }

    fn argmax(q: &[f32]) -> usize {
        let mut best = 0;
        for a in 1..ACTION_COUNT {
            if q[a] > q[best] {
                best = a;
            }
        }
        best
    }

    fn convert(dst: &mut [f32], src: &[f64]) -> Result<()> {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = to_f32(s)?;
        }
        Ok(())
    }

    fn train_batch(&mut self) -> Result<()> {
        let n = self.config.batch_size;
        let width = self.state_size;
        for b in 0..n {
            let i = self.rng.index(self.stored);
            self.batch_index[b] = i;
            self.batch_states[b * width..(b + 1) * width].copy_from_slice(&self.states[i * width..(i + 1) * width]);
            self.batch_next[b * width..(b + 1) * width].copy_from_slice(&self.next_states[i * width..(i + 1) * width]);
        }
        // Double DQN: the online net picks the action, the target net scores
        // it. Weights do not move inside a batch, so the passes can be grouped.
        let picks: Vec<usize> = self
            .online
            .forward(&self.batch_next, n)?
            .as_chunks::<ACTION_COUNT>()
            .0
            .iter()
            .map(|q| Self::argmax(q))
            .collect();
        let scored = self.target.forward(&self.batch_next, n)?;
        for b in 0..n {
            let i = self.batch_index[b];
            self.batch_target[b] = self.rewards[i];
            if !self.dones[i] {
                self.batch_target[b] += self.config.discount * f64::from(scored[b * ACTION_COUNT + picks[b]]);
            }
        }
        let q = self.online.forward(&self.batch_states, n)?;
        let d = self.config.huber_delta;
        let mut batch_loss = 0.0;
        let mut grads = std::mem::take(&mut self.batch_grad);
        grads.fill(0.0);
        for b in 0..n {
            let a = self.actions[self.batch_index[b]] as usize;
            let error = f64::from(q[b * ACTION_COUNT + a]) - self.batch_target[b];
            // Huber, so one violent bar cannot dominate the update.
            grads[b * ACTION_COUNT + a] = to_f32(clamp(error, -d, d))?;
            batch_loss += if error.abs() <= d {
                0.5 * error * error
            } else {
                d * (error.abs() - 0.5 * d)
            };
        }
        let result = self.online.backward(&grads, n).and_then(|_| self.online.step());
        self.batch_grad = grads;
        result?;
        self.loss += (batch_loss / n as f64 - self.loss) * 0.01;
        Ok(())
    }
}

impl Agent for Dqn {
    fn name(&self) -> &str {
        "dqn"
    }

    fn act(&mut self, state: &[f64], greedy: bool) -> Result<usize> {
        if !greedy && self.rng.uniform() < self.exploration() {
            return Ok(self.rng.index(ACTION_COUNT));
        }
        Self::convert(&mut self.input, state)?;
        let out = self.online.forward(&self.input, 1)?;
        let q = [f64::from(out[0]), f64::from(out[1]), f64::from(out[2])];
        let mut best = 0;
        for a in 1..ACTION_COUNT {
            if q[a] > q[best] {
                best = a;
            }
        }
        if self.config.switch_margin <= 0.0 || state.len() < 3 {
            return Ok(best);
        }
        let position = held(state);
        if !(-1..=1).contains(&position) {
            return Ok(best);
        }
        let held = position_to_action(position);
        if best == held {
            return Ok(best);
        }
        let moved = f64::from((action_to_position(best) - action_to_position(held)).abs());
        Ok(if q[best] - q[held] < self.config.switch_margin * moved {
            held
        } else {
            best
        })
    }

    fn observe(&mut self, state: &[f64], action: usize, reward: f64, next: &[f64], done: bool) -> Result<()> {
        let width = self.state_size;
        let base = self.cursor * width;
        Self::convert(&mut self.states[base..base + width], state)?;
        Self::convert(&mut self.next_states[base..base + width], next)?;
        self.actions[self.cursor] = action as u8;
        self.rewards[self.cursor] = reward * self.config.reward_scale;
        self.dones[self.cursor] = done;
        self.cursor = (self.cursor + 1) % self.config.replay_capacity;
        self.stored = (self.stored + 1).min(self.config.replay_capacity);
        self.steps += 1;
        if self.stored >= self.config.replay_warmup && self.steps % self.config.train_every as i64 == 0 {
            self.train_batch()?;
        }
        if self.steps % self.config.target_sync as i64 == 0 {
            self.target.copy_from(&self.online)?;
        }
        Ok(())
    }

    fn end_epoch(&mut self) -> Result<()> {
        if self.config.lr_decay != 1.0 {
            self.online.scale_learning_rate(to_f32(self.config.lr_decay)?)?;
        }
        Ok(())
    }

    fn learns(&self) -> bool {
        true
    }

    fn exploration(&self) -> f64 {
        decayed(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.steps,
            self.config.epsilon_decay_steps,
        )
    }

    fn recent_loss(&self) -> f64 {
        self.loss
    }

    fn policy(&self) -> Option<(&'static str, Vec<u8>)> {
        Some(("weights.bin", self.online.to_bytes()))
    }
}

/// Votes the greedy actions of independently seeded learners. Averaging
/// seeds is the cheapest variance reduction there is, and it removes the
/// need to pick one seed's policy off the test window.
pub struct Ensemble<'a> {
    members: Vec<&'a mut dyn Agent>,
}

impl<'a> Ensemble<'a> {
    pub fn new(members: Vec<&'a mut dyn Agent>) -> Ensemble<'a> {
        Ensemble { members }
    }
}

impl Agent for Ensemble<'_> {
    fn name(&self) -> &str {
        "ensemble"
    }

    fn act(&mut self, state: &[f64], greedy: bool) -> Result<usize> {
        let mut votes = [0usize; ACTION_COUNT];
        for member in &mut self.members {
            votes[member.act(state, greedy)?] += 1;
        }
        let mut best = 0;
        for a in 1..ACTION_COUNT {
            if votes[a] > votes[best] {
                best = a;
            }
        }
        // A split vote is not an opinion: holding is what the members
        // disagreed about, so changing on a tie would manufacture turnover.
        if votes.iter().filter(|&&v| v == votes[best]).count() > 1 {
            return Ok(position_to_action(held(state)));
        }
        Ok(best)
    }

    fn begin_episode(&mut self) {
        for member in &mut self.members {
            member.begin_episode();
        }
    }
}

/// Build the named agent.
pub fn make_agent(
    name: &str,
    features: &FeatureMatrix,
    dqn: &DqnConfig,
    tabular: &TabularConfig,
    seed: u64,
) -> Result<Box<dyn Agent>> {
    Ok(match name {
        "flat" => Box::new(Fixed {
            action: position_to_action(0),
            label: "flat",
        }),
        "long" => Box::new(Fixed {
            action: position_to_action(1),
            label: "long",
        }),
        "short" => Box::new(Fixed {
            action: position_to_action(-1),
            label: "short",
        }),
        "random" => Box::new(RandomAgent {
            flip: 0.02,
            rng: Rng::new(seed),
            held: 1,
        }),
        "crossover" => Box::new(Crossover {
            threshold: 0.5,
            trend_column: features.column("trend_fast")?,
        }),
        "breakout" => Box::new(Breakout {
            threshold: 0.0,
            high_column: features.column("pd_high_dist")?,
        }),
        "qtable" => Box::new(TabularQ::new(tabular, features, seed)?),
        "dqn" => Box::new(Dqn::new(dqn, features.columns + 3, seed)?),
        _ => {
            return Err(invalid(format!(
                "unknown agent '{name}'; choose one of {}",
                AGENT_NAMES.join(", ")
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::build_features;
    use crate::synthetic;

    fn state(features: usize, position: f64) -> Vec<f64> {
        let mut s = vec![0.0; features + 3];
        s[features] = position;
        s
    }

    #[test]
    fn rules_hold_inside_their_bands() {
        let f = build_features(&synthetic::trend_series(2000)).unwrap();
        let dqn = DqnConfig::default();
        let tab = TabularConfig::default();
        let mut cross = make_agent("crossover", &f, &dqn, &tab, 1).unwrap();
        let trend = f.column("trend_fast").unwrap();
        let mut s = state(f.columns, 1.0);
        assert_eq!(cross.act(&s, true).unwrap(), 2);
        s[trend] = -0.6;
        assert_eq!(cross.act(&s, true).unwrap(), 0);
        let mut breakout = make_agent("breakout", &f, &dqn, &tab, 1).unwrap();
        let high = f.column("pd_high_dist").unwrap();
        let mut s = state(f.columns, 0.0);
        s[high] = 0.1;
        assert_eq!(breakout.act(&s, true).unwrap(), 2);
        s[high] = 0.0;
        assert_eq!(breakout.act(&s, true).unwrap(), 1);
        assert!(make_agent("nope", &f, &dqn, &tab, 1).is_err());
    }

    #[test]
    fn ensemble_ties_hold() {
        let mut long = Fixed {
            action: 2,
            label: "long",
        };
        let mut short = Fixed {
            action: 0,
            label: "short",
        };
        let mut e = Ensemble::new(vec![&mut long, &mut short]);
        assert_eq!(e.act(&[0.0, 0.0, 0.0, 0.0], true).unwrap(), 1);
        assert_eq!(e.act(&[0.0, -1.0, 0.0, 0.0], true).unwrap(), 0);
    }

    #[test]
    fn dqn_trains_deterministically() {
        let f = build_features(&synthetic::trend_series(2000)).unwrap();
        let config = DqnConfig {
            hidden: vec![8, 8],
            replay_capacity: 500,
            replay_warmup: 50,
            batch_size: 8,
            target_sync: 40,
            ..DqnConfig::default()
        };
        let run = || {
            let mut agent = Dqn::new(&config, f.columns + 3, 9).unwrap();
            let mut s = state(f.columns, 0.0);
            for t in 0..400 {
                s[0] = (t as f64 * 0.1).sin();
                let a = agent.act(&s, false).unwrap();
                let mut next = s.clone();
                next[f.columns] = action_to_position(a) as f64;
                agent.observe(&s, a, s[0] * 1e-4, &next, t % 50 == 49).unwrap();
                s = next;
            }
            agent.end_epoch().unwrap();
            (agent.policy().unwrap().1, agent.recent_loss())
        };
        let (a, loss) = run();
        let (b, _) = run();
        assert_eq!(a, b);
        assert!(loss > 0.0 && loss.is_finite());
    }

    #[test]
    fn qtable_policy_lists_every_state() {
        let f = build_features(&synthetic::trend_series(2000)).unwrap();
        let q = TabularQ::new(&TabularConfig::default(), &f, 1).unwrap();
        let (name, bytes) = q.policy().unwrap();
        assert_eq!(name, "policy.csv");
        assert_eq!(String::from_utf8(bytes).unwrap().lines().count(), 226);
        assert!(
            TabularQ::new(
                &TabularConfig {
                    learning_rate: 0.0,
                    ..TabularConfig::default()
                },
                &f,
                1
            )
            .is_err()
        );
    }
}
