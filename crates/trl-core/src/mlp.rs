//! A small multilayer perceptron: binary32 weights, ReLU hidden layers, a
//! linear output, He/Glorot-uniform initialisation from SplitMix64, summed
//! (unaveraged) gradients and an Adam step computed in binary64 with
//! global-norm clipping.
//!
//! Every reduction runs in a fixed order (dot products in eight lanes, then a
//! fixed pairwise sum), never fused into FMAs, so training reproduces
//! bit-exactly on every platform and optimisation level.

use crate::error::{Result, invalid};

/// Recorded in manifests: a run trained under other numerics is not reproducible here.
pub const DQN_NUMERICS: &str = "trl-mlp-f32-lanes8-he-splitmix64-v1";

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;
const GRADIENT_CLIP: f32 = 10.0;
const MAGIC: &[u8; 8] = b"TRLMLP01";

fn numeric() -> crate::Error {
    invalid("the network produced a non-finite value; lower --lr or check the inputs")
}

/// Convert at the network boundary: the experiment is binary64, the network binary32.
pub fn to_f32(v: f64) -> Result<f32> {
    if !v.is_finite() || v.abs() > f64::from(f32::MAX) {
        return Err(invalid("network input must be finite and within binary32 range"));
    }
    let c = v as f32;
    if v != 0.0 && c == 0.0 {
        return Err(invalid("network input would underflow to zero"));
    }
    Ok(c)
}

/// Mix a 64-bit seed into the network's nonzero 32-bit seed.
pub fn network_seed(seed: u64) -> u32 {
    let mut s = seed;
    s ^= s >> 30;
    s = s.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    s ^= s >> 27;
    s = s.wrapping_mul(0x94d0_49bb_1331_11eb);
    s ^= s >> 31;
    let r = (s ^ (s >> 32)) as u32;
    if r == 0 { 1 } else { r }
}

struct InitRng(u64);

impl InitRng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 32) as u32
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut lanes = [0f32; 8];
    let (ca, ta) = a.as_chunks::<8>();
    let (cb, tb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            lanes[l] += x[l] * y[l];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ta.iter().zip(tb) {
        tail += x * y;
    }
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7])) + tail
}

/// `dst += src * scale`, elementwise.
#[inline]
fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s * scale;
    }
}

#[derive(Clone, Copy, Debug)]
struct Layer {
    inputs: usize,
    outputs: usize,
    weight: usize,
    bias: usize,
    /// Offset of the layer's input in a sample's activation row.
    input: usize,
    output: usize,
}

#[derive(Clone, Debug)]
pub struct Mlp {
    widths: Vec<usize>,
    layers: Vec<Layer>,
    stride: usize,
    params: Vec<f32>,
    grads: Vec<f32>,
    moment1: Vec<f32>,
    moment2: Vec<f32>,
    act: Vec<f32>,
    delta: Vec<f32>,
    out: Vec<f32>,
    batch: usize,
    accumulated: u64,
    steps: u64,
    learning_rate: f32,
}

impl Mlp {
    /// A network with the given layer widths (inputs first, outputs last).
    pub fn new(widths: &[usize], seed: u32) -> Result<Mlp> {
        if widths.len() < 2 || widths.contains(&0) {
            return Err(invalid("a network needs at least two layers of positive width"));
        }
        let mut layers = Vec::with_capacity(widths.len() - 1);
        let (mut parameters, mut stride) = (0usize, 0usize);
        for w in widths.windows(2) {
            let (inputs, outputs) = (w[0], w[1]);
            let layer = Layer {
                inputs,
                outputs,
                weight: parameters,
                bias: parameters + inputs * outputs,
                input: stride,
                output: stride + inputs,
            };
            parameters = layer.bias + outputs;
            stride += inputs;
            layers.push(layer);
        }
        stride += widths[widths.len() - 1];
        let mut params = vec![0f32; parameters];
        let mut rng = InitRng(u64::from(seed));
        let count = layers.len();
        for (i, layer) in layers.iter().enumerate() {
            let relu = i + 1 < count;
            let limit = if relu {
                (6.0f32 / layer.inputs as f32).sqrt()
            } else {
                (6.0f32 / (layer.inputs as f32 + layer.outputs as f32)).sqrt()
            };
            for p in &mut params[layer.weight..layer.bias] {
                let uniform = (rng.next() >> 8) as f32 * (1.0f32 / 16_777_216.0);
                *p = (2.0f32 * uniform - 1.0f32) * limit;
            }
        }
        Ok(Mlp {
            widths: widths.to_vec(),
            layers,
            stride,
            grads: vec![0.0; parameters],
            moment1: vec![0.0; parameters],
            moment2: vec![0.0; parameters],
            params,
            act: Vec::new(),
            delta: Vec::new(),
            out: Vec::new(),
            batch: 0,
            accumulated: 0,
            steps: 0,
            learning_rate: 1e-3,
        })
    }

    pub fn inputs(&self) -> usize {
        self.widths[0]
    }

    pub fn outputs(&self) -> usize {
        self.widths[self.widths.len() - 1]
    }

    pub fn parameter_count(&self) -> usize {
        self.params.len()
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, rate: f32) -> Result<()> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(invalid("the learning rate must be positive"));
        }
        self.learning_rate = rate;
        Ok(())
    }

    pub fn scale_learning_rate(&mut self, factor: f32) -> Result<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(invalid("the learning-rate factor must be positive"));
        }
        let rate = self.learning_rate * factor;
        if !rate.is_finite() || rate <= 0.0 {
            return Err(numeric());
        }
        self.learning_rate = rate;
        Ok(())
    }

    /// Run `samples` input rows; returns their output rows back to back.
    pub fn forward(&mut self, input: &[f32], samples: usize) -> Result<&[f32]> {
        self.batch = 0;
        let width = self.inputs();
        if samples == 0 || input.len() != samples * width {
            return Err(invalid("network input width mismatch"));
        }
        if !input.iter().all(|v| v.is_finite()) {
            return Err(numeric());
        }
        let stride = self.stride;
        if self.act.len() < samples * stride {
            self.act.resize(samples * stride, 0.0);
            self.delta.resize(samples * stride, 0.0);
        }
        for s in 0..samples {
            self.act[s * stride..s * stride + width].copy_from_slice(&input[s * width..(s + 1) * width]);
        }
        let count = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            let relu = i + 1 < count;
            for o in 0..layer.outputs {
                let row = &self.params[layer.weight + o * layer.inputs..layer.weight + (o + 1) * layer.inputs];
                let bias = self.params[layer.bias + o];
                for s in 0..samples {
                    let base = s * stride;
                    let z = dot(row, &self.act[base + layer.input..base + layer.input + layer.inputs]) + bias;
                    if !z.is_finite() {
                        return Err(numeric());
                    }
                    // z is finite here, so this is ReLU with a derivative of 0 at 0.
                    self.act[base + layer.output + o] = if relu && z <= 0.0 { 0.0 } else { z };
                }
            }
        }
        let last = self.layers[count - 1];
        self.out.clear();
        for s in 0..samples {
            let base = s * stride + last.output;
            self.out.extend_from_slice(&self.act[base..base + last.outputs]);
        }
        self.batch = samples;
        Ok(&self.out)
    }

    /// Accumulate the gradient of the batch just run, given dL/d(output) per
    /// sample (not averaged; the step divides by the samples accumulated).
    pub fn backward(&mut self, grad_output: &[f32], samples: usize) -> Result<()> {
        if self.batch == 0 || samples != self.batch {
            return Err(invalid("network backward requires its retained forward batch"));
        }
        let outputs = self.outputs();
        if grad_output.len() != samples * outputs || !grad_output.iter().all(|v| v.is_finite()) {
            return Err(numeric());
        }
        let stride = self.stride;
        let last = self.layers[self.layers.len() - 1];
        for s in 0..samples {
            let base = s * stride + last.output;
            self.delta[base..base + outputs].copy_from_slice(&grad_output[s * outputs..(s + 1) * outputs]);
        }
        self.batch = 0;
        let result = self.backward_deltas(samples);
        if result.is_err() {
            self.zero_grad();
        }
        result
    }

    fn backward_deltas(&mut self, samples: usize) -> Result<()> {
        let stride = self.stride;
        for i in (0..self.layers.len()).rev() {
            let layer = self.layers[i];
            for o in 0..layer.outputs {
                let mut bias_gradient = self.grads[layer.bias + o];
                for s in 0..samples {
                    let base = s * stride;
                    let delta = self.delta[base + layer.output + o];
                    if delta == 0.0 {
                        continue;
                    }
                    let row = &mut self.grads[layer.weight + o * layer.inputs..layer.weight + (o + 1) * layer.inputs];
                    axpy(
                        row,
                        &self.act[base + layer.input..base + layer.input + layer.inputs],
                        delta,
                    );
                    bias_gradient += delta;
                }
                if !bias_gradient.is_finite() {
                    return Err(numeric());
                }
                self.grads[layer.bias + o] = bias_gradient;
            }
            if i == 0 {
                break;
            }
            for s in 0..samples {
                let base = s * stride + layer.input;
                self.delta[base..base + layer.inputs].fill(0.0);
            }
            for o in 0..layer.outputs {
                let row = &self.params[layer.weight + o * layer.inputs..layer.weight + (o + 1) * layer.inputs];
                for s in 0..samples {
                    let base = s * stride;
                    let scale = self.delta[base + layer.output + o];
                    axpy(
                        &mut self.delta[base + layer.input..base + layer.input + layer.inputs],
                        row,
                        scale,
                    );
                }
            }
            for s in 0..samples {
                let base = s * stride + layer.input;
                for k in 0..layer.inputs {
                    if self.act[base + k] <= 0.0 {
                        self.delta[base + k] = 0.0;
                    }
                }
                if !self.delta[base..base + layer.inputs].iter().all(|v| v.is_finite()) {
                    return Err(numeric());
                }
            }
        }
        self.accumulated += samples as u64;
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.grads.fill(0.0);
        self.accumulated = 0;
    }

    /// One Adam step on the mean of the accumulated gradients, clipped to a
    /// global norm of 10.
    pub fn step(&mut self) -> Result<()> {
        if self.accumulated == 0 {
            return Err(invalid("no gradient to step on"));
        }
        let mut scale = 1.0 / self.accumulated as f64;
        let clip = f64::from(GRADIENT_CLIP);
        let mut norm = 0.0f64;
        for &g in &self.grads {
            let g = f64::from(g) * scale;
            norm += g * g;
        }
        let norm = norm.sqrt();
        if norm > clip {
            scale *= clip / norm;
        }
        let (beta1, beta2) = (f64::from(BETA1), f64::from(BETA2));
        let steps = (self.steps + 1) as f64;
        let bc1 = 1.0 - crate::math::pow(beta1, steps);
        let bc2 = 1.0 - crate::math::pow(beta2, steps);
        let rate = f64::from(self.learning_rate);
        let epsilon = f64::from(EPSILON);
        let limit = f64::from(f32::MAX);
        let mut next = Vec::with_capacity(self.params.len());
        for i in 0..self.params.len() {
            let g = f64::from(self.grads[i]) * scale;
            let m = beta1 * f64::from(self.moment1[i]) + (1.0 - beta1) * g;
            let v = beta2 * f64::from(self.moment2[i]) + (1.0 - beta2) * g * g;
            let p = f64::from(self.params[i]) - rate * (m / bc1) / ((v / bc2).sqrt() + epsilon);
            if !m.is_finite() || !v.is_finite() || !p.is_finite() || m.abs() > limit || v > limit || p.abs() > limit {
                return Err(numeric());
            }
            next.push((m as f32, v as f32, p as f32));
        }
        for (i, (m, v, p)) in next.into_iter().enumerate() {
            self.moment1[i] = m;
            self.moment2[i] = v;
            self.params[i] = p;
        }
        self.steps += 1;
        self.batch = 0;
        self.zero_grad();
        Ok(())
    }

    /// Copy `source`'s weights; optimiser state and gradients reset.
    pub fn copy_from(&mut self, source: &Mlp) -> Result<()> {
        if self.widths != source.widths {
            return Err(invalid("networks differ in shape"));
        }
        self.params.copy_from_slice(&source.params);
        self.moment1.fill(0.0);
        self.moment2.fill(0.0);
        self.steps = 0;
        self.batch = 0;
        self.zero_grad();
        Ok(())
    }

    /// The checkpoint written as `weights.bin`: little-endian, never a struct dump.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.params.len() * 12);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.widths.len() as u32).to_le_bytes());
        for &w in &self.widths {
            out.extend_from_slice(&(w as u32).to_le_bytes());
        }
        out.extend_from_slice(&self.learning_rate.to_le_bytes());
        out.extend_from_slice(&self.steps.to_le_bytes());
        for v in self.params.iter().chain(&self.moment1).chain(&self.moment2) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Read a checkpoint written by [`Mlp::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Mlp> {
        let bad = || invalid("not a trade-rl network checkpoint");
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = bytes.get(at..at + n).ok_or_else(bad)?;
            at += n;
            Ok(s)
        };
        if take(8)? != MAGIC {
            return Err(bad());
        }
        let layers = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        if !(2..=64).contains(&layers) {
            return Err(bad());
        }
        let mut widths = Vec::with_capacity(layers);
        for _ in 0..layers {
            widths.push(u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize);
        }
        let mut mlp = Mlp::new(&widths, 1)?;
        mlp.learning_rate = f32::from_le_bytes(take(4)?.try_into().unwrap());
        mlp.steps = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let n = mlp.params.len();
        for target in [&mut mlp.params, &mut mlp.moment1, &mut mlp.moment2] {
            let raw = take(n * 4)?;
            for (v, c) in target.iter_mut().zip(raw.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*c);
            }
        }
        if at != bytes.len() {
            return Err(bad());
        }
        Ok(mlp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialisation_is_bounded_and_seeded() {
        let a = Mlp::new(&[4, 8, 3], 7).unwrap();
        let b = Mlp::new(&[4, 8, 3], 7).unwrap();
        let c = Mlp::new(&[4, 8, 3], 8).unwrap();
        assert_eq!(a.params, b.params);
        assert_ne!(a.params, c.params);
        assert_eq!(a.parameter_count(), 4 * 8 + 8 + 8 * 3 + 3);
        let hidden_limit = (6.0f32 / 4.0).sqrt();
        assert!(a.params[..32].iter().all(|w| w.abs() <= hidden_limit));
        assert!(a.params[32..40].iter().all(|&b| b == 0.0));
        assert_eq!(network_seed(0), network_seed(0));
        assert_ne!(network_seed(0), 0);
    }

    #[test]
    fn dot_matches_naive_within_rounding() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.11).cos()).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        assert!((f64::from(dot(&a, &b)) - naive).abs() < 1e-5);
    }

    #[test]
    fn learns_a_linear_target() {
        let mut net = Mlp::new(&[2, 16, 1], 3).unwrap();
        net.set_learning_rate(1e-2).unwrap();
        let xs: Vec<[f32; 2]> = (0..64).map(|i| [(i % 8) as f32 / 8.0, (i / 8) as f32 / 8.0]).collect();
        let target = |x: &[f32; 2]| 2.0 * x[0] - x[1] + 0.5;
        let mut first = None;
        let mut last = 0.0;
        for _ in 0..400 {
            let input: Vec<f32> = xs.iter().flatten().copied().collect();
            let out = net.forward(&input, xs.len()).unwrap().to_vec();
            let mut loss = 0.0;
            let grads: Vec<f32> = out
                .iter()
                .zip(&xs)
                .map(|(y, x)| {
                    let e = y - target(x);
                    loss += e * e;
                    e
                })
                .collect();
            first.get_or_insert(loss);
            last = loss;
            net.backward(&grads, xs.len()).unwrap();
            net.step().unwrap();
        }
        assert!(last < first.unwrap() / 50.0, "{first:?} -> {last}");
    }

    #[test]
    fn checkpoint_round_trip_and_copy() {
        let mut a = Mlp::new(&[3, 5, 2], 11).unwrap();
        a.forward(&[0.1, -0.2, 0.3], 1).unwrap();
        a.backward(&[0.5, -0.5], 1).unwrap();
        a.step().unwrap();
        let back = Mlp::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(back.params, a.params);
        assert_eq!(back.to_bytes(), a.to_bytes());
        let mut b = Mlp::new(&[3, 5, 2], 12).unwrap();
        b.copy_from(&a).unwrap();
        assert_eq!(b.params, a.params);
        assert!(b.moment1.iter().all(|&m| m == 0.0));
        assert!(Mlp::new(&[3, 4, 2], 1).unwrap().copy_from(&a).is_err());
    }

    #[test]
    fn conversions_are_checked() {
        assert!(to_f32(f64::NAN).is_err());
        assert!(to_f32(1e300).is_err());
        assert!(to_f32(1e-300).is_err());
        assert_eq!(to_f32(0.0).unwrap(), 0.0);
        assert_eq!(to_f32(0.5).unwrap(), 0.5);
    }
}
