/// small extension of rust-ndarray for convenience
extern crate ndarray;

/// type alias for convenience
pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

/// type alias for convenience
pub type NdArrayView<'a> = ndarray::ArrayView<'a, f32, ndarray::IxDyn>;

/// exposes array_gen
pub use array_gen::*;

#[inline]
pub fn arr_to_shape(arr: &NdArray) -> Vec<usize>
{
    arr.iter().map(|&a| a as usize).collect::<Vec<_>>()
}

#[doc(hidden)]
#[inline]
// TODO: remove unwrap
pub fn expand_dims_view<'a>(x: NdArrayView<'a>, axis: usize) -> NdArrayView<'a>
{
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
// TODO: remove unwrap
pub fn expand_dims(x: NdArray, axis: usize) -> NdArray
{
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn roll_axis(arg: &mut NdArray, to: ndarray::Axis, from: ndarray::Axis)
{
    let i = to.index();
    let mut j = from.index();
    if j > i {
        while i != j {
            arg.swap_axes(i, j);
            j -= 1;
        }
    } else {
        while i != j {
            arg.swap_axes(i, j);
            j += 1;
        }
    }
}

#[inline]
pub fn normalize_negative_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

#[inline]
pub fn normalize_negative_axes(axes: &NdArray, ndim: usize) -> Vec<usize> {
    let mut axes_ret: Vec<usize> = Vec::with_capacity(axes.len());
    for &axis in axes.iter() {
        let axis = if axis < 0. {
            (ndim as f32 + axis) as usize
        } else {
            axis as usize
        };
        axes_ret.push(axis);
    }
    axes_ret
}

#[inline]
pub fn sparse_to_dense(arr: &NdArray) -> Vec<usize> {
    let mut axes: Vec<usize> = vec![];
    for (i, &a) in arr.iter().enumerate() {
        if a == 1. {
            axes.push(i as usize);
        }
    }
    axes
}

#[doc(hidden)]
#[inline]
/// This works well only for small array
pub fn vec_as_shape(x: &NdArray) -> Vec<usize>
{
    let mut target = Vec::with_capacity(x.len());
    for &a in x.iter() {
        target.push(a as usize);
    }
    target
}

#[doc(hidden)]
#[inline]
pub fn scalar_shape() -> NdArray
{
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn is_scalar_shape(shape: &[usize]) -> bool {
    shape == &[] || shape == &[0]
}

#[doc(hidden)]
#[inline]
pub fn shape_of(x: &NdArray) -> NdArray
{
    let shape = x.shape().iter().map(|&a| a as f32).collect::<Vec<f32>>();
    let rank = shape.len();
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[rank]), shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn into_mat(x: NdArray) -> ndarray::Array<f32, ndarray::Ix2>
{
    let (a, b) = {
        let shape = x.shape();
        (shape[0], shape[1])
    };
    x.into_shape(ndarray::Ix2(a, b)).unwrap()
}

/// Generates ndarray which can be fed to `autograd::variable()` etc.
pub mod array_gen
{
    use super::*;

    #[inline]
    /// Zeros.
    pub fn zeros(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        NdArray::from_elem(shape, 0.)
    }

    #[inline]
    /// Ones.
    pub fn ones(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        NdArray::from_elem(shape, 1.)
    }

    #[inline]
    /// Create ndarray object from a scalar.
    pub fn from_scalar(val: f32) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        NdArray::from_elem(ndarray::IxDyn(&[]), val)
    }

    #[inline]
    /// Permutation.
    pub fn permutation(size: usize) -> ndarray::Array1<usize>
    {
        ArrRng::default().permutation(size)
    }

    #[inline]
    /// Samples from normal distribution
    pub fn random_normal(
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().random_normal(shape, mean, stddev)
    }

    #[inline]
    /// Samples from uniform distribution.
    pub fn random_uniform(
        shape: &[usize],
        min: f64,
        max: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().random_uniform(shape, min, max)
    }

    #[inline]
    /// Samples from standard normal distribution
    pub fn standard_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().standard_normal(shape)
    }

    #[inline]
    /// Samples from standard uniform distribution
    pub fn standard_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().standard_uniform(shape)
    }

    #[inline]
    /// Glorot normal initialization. (a.k.a. Xavier normal initialization)
    pub fn glorot_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().glorot_normal(shape)
    }

    #[inline]
    /// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
    pub fn glorot_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().glorot_normal(shape)
    }

    /// Bernoulli distribution.
    #[inline]
    pub fn bernoulli(shape: &[usize], p: f64) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().bernoulli(shape, p)
    }

    /// Exponential distribution.
    #[inline]
    pub fn exponential(shape: &[usize], lambda: f64) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().exponential(shape, lambda)
    }

    /// Log normal distribution.
    #[inline]
    pub fn log_normal(
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().log_normal(shape, mean, stddev)
    }

    /// Gamma distribution.
    #[inline]
    pub fn gamma(
        shape: &[usize],
        shape_param: f64,
        scale: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        ArrRng::default().gamma(shape, shape_param, scale)
    }
}

use std::cell::RefCell;
use rand::{self, Rng, XorShiftRng};
use rand::distributions::IndependentSample;

pub struct ArrRng<R = XorShiftRng> {
    rng: RefCell<R>
}

impl Default for ArrRng<XorShiftRng>
{
    fn default() -> Self
    {
        ArrRng {
            rng: RefCell::new(rand::weak_rng())
        }
    }
}

impl<R> ArrRng<R> {
    pub fn new(rng: R) -> Self
    {
        ArrRng {
            rng: RefCell::new(rng)
        }
    }
}

impl<R: Rng> ArrRng<R>
{
    #[inline]
    fn gen_rnd_array<T>(&self, shape: &[usize], dist: T) -> NdArray
    where
        T: IndependentSample<f64>,
    {
        let mut rng = self.rng.borrow_mut();
        NdArray::from_shape_fn(shape, |_| dist.ind_sample(&mut *rng) as f32)
    }

    #[inline]
    fn gen_rand_array_f<T, F>(&self, shape: &[usize], dist: T, f: F) -> NdArray
    where
        T: IndependentSample<f64>,
        F: Fn(f64) -> f64,
    {
        let mut rng = self.rng.borrow_mut();
        NdArray::from_shape_fn(shape, |_| f(dist.ind_sample(&mut *rng)) as f32)
    }

    #[inline]
    /// Permutation.
    pub fn permutation(&mut self, size: usize) -> ndarray::Array1<usize>
    {
        let mut data: Vec<usize> = (0..size).collect();
        let slice = data.as_mut_slice();

        let mut rng = self.rng.borrow_mut();
        rng.shuffle(slice);
        ndarray::Array1::<usize>::from_vec(slice.to_vec())
    }

    #[inline]
    /// Samples from normal distribution
    pub fn random_normal(
        &self,
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let normal = rand::distributions::Normal::new(mean, stddev);
        self.gen_rnd_array(shape, normal)
    }

    #[inline]
    /// Samples from uniform distribution.
    pub fn random_uniform(
        &self,
        shape: &[usize],
        min: f64,
        max: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let range = rand::distributions::Range::new(min, max);
        self.gen_rnd_array(shape, range)
    }

    #[inline]
    /// Samples from standard normal distribution
    pub fn standard_normal(&self, shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let normal = rand::distributions::Normal::new(0., 1.);
        self.gen_rnd_array(shape, normal)
    }

    #[inline]
    /// Samples from standard uniform distribution
    pub fn standard_uniform(&self, shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Range::new(0., 1.);
        self.gen_rnd_array(shape, dist)
    }

    #[inline]
    /// Glorot normal initialization. (a.k.a. Xavier normal initialization)
    pub fn glorot_normal(&self, shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        assert_eq!(shape.len(), 2);
        let s = 1. / (shape[0] as f64).sqrt();
        let normal = rand::distributions::Normal::new(0., s);
        self.gen_rnd_array(shape, normal)
    }

    #[inline]
    /// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
    pub fn glorot_uniform(&self, shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        assert_eq!(shape.len(), 2);
        let s = (6. / shape[0] as f64).sqrt();
        let uniform = rand::distributions::Range::new(-s, s);
        self.gen_rnd_array(shape, uniform)
    }

    /// Bernoulli distribution.
    #[inline]
    pub fn bernoulli(&self, shape: &[usize], p: f64) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Range::new(0., 1.);
        self.gen_rand_array_f(shape, dist, |a| (a < p) as i64 as f64)
    }

    /// Exponential distribution.
    #[inline]
    pub fn exponential(&self, shape: &[usize], lambda: f64) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Exp::new(lambda);
        self.gen_rnd_array(shape, dist)
    }

    /// Log normal distribution.
    #[inline]
    pub fn log_normal(
        &self,
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::LogNormal::new(mean, stddev);
        self.gen_rnd_array(shape, dist)
    }

    /// Gamma distribution.
    #[inline]
    pub fn gamma(
        &self,
        shape: &[usize],
        shape_param: f64,
        scale: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Gamma::new(shape_param, scale);
        self.gen_rnd_array(shape, dist)
    }

}
