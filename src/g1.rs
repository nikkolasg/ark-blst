use ark_ec::{
    models::CurveConfig,
    scalar_mul::{variable_base::VariableBaseMSM, ScalarMul},
    AffineRepr, CurveGroup, Group,
};
use ark_ff::Zero;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    io::{Read, Write},
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
};
use blstrs::{impl_add_sub, impl_add_sub_assign, impl_mul, impl_mul_assign};
use core::{
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Neg, Sub, SubAssign},
};
use group::{prime::PrimeCurveAffine, Curve as _, Group as _};
use rand::SeedableRng;
use zeroize::Zeroize;

use crate::fp::Fp;
use crate::scalar::Scalar;

const COMPRESSED_SIZE: usize = 48;
const UNCOMPRESSED_SIZE: usize = 96;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Config;

impl CurveConfig for Config {
    type BaseField = Fp;
    type ScalarField = Scalar;

    /// COFACTOR = (x - 1)^2 / 3  = 76329603384216526031706109802092473003
    const COFACTOR: &'static [u64] = &[0x8c00aaab0000aaab, 0x396c8c005555e156];

    /// COFACTOR_INV = COFACTOR^{-1} mod r
    /// = 52435875175126190458656871551744051925719901746859129887267498875565241663483
    const COFACTOR_INV: Scalar = Scalar(blstrs::Scalar::from_raw_unchecked([
        288839107172787499,
        1152722415086798946,
        2612889808468387987,
        5124657601728438008,
    ]));
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct G1Affine(blstrs::G1Affine);

impl Deref for G1Affine {
    type Target = blstrs::G1Affine;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO check invariant
#[allow(unknown_lints, renamed_and_removed_lints)]
#[allow(clippy::derived_hash_with_manual_eq, clippy::derive_hash_xor_eq)]
impl Hash for G1Affine {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_uncompressed()[..]);
    }
}

impl Neg for G1Affine {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.0 = -self.0;
        self
    }
}

impl Add<&G1Affine> for G1Affine {
    type Output = G1Projective;

    #[inline]
    fn add(self, rhs: &G1Affine) -> Self::Output {
        G1Projective(self.0 + blstrs::G1Projective::from(rhs.0))
    }
}

impl Add<G1Affine> for G1Affine {
    type Output = G1Projective;

    #[inline]
    fn add(self, rhs: G1Affine) -> Self::Output {
        G1Projective(self.0 + blstrs::G1Projective::from(&rhs.0))
    }
}

impl Add<&G1Projective> for &G1Projective {
    type Output = G1Projective;

    #[inline]
    fn add(self, rhs: &G1Projective) -> G1Projective {
        G1Projective(self.0 + rhs.0)
    }
}

impl Add<&G1Projective> for &G1Affine {
    type Output = G1Projective;

    #[inline]
    fn add(self, rhs: &G1Projective) -> G1Projective {
        G1Projective(self.0 + rhs.0)
    }
}

impl Add<&G1Affine> for &G1Projective {
    type Output = G1Projective;

    #[inline]
    fn add(self, rhs: &G1Affine) -> G1Projective {
        G1Projective(self.0 + rhs.0)
    }
}

impl Sub<&G1Projective> for &G1Projective {
    type Output = G1Projective;

    #[inline]
    fn sub(self, rhs: &G1Projective) -> G1Projective {
        G1Projective(self.0 - rhs.0)
    }
}

impl Sub<&G1Projective> for &G1Affine {
    type Output = G1Projective;

    #[inline]
    fn sub(self, rhs: &G1Projective) -> G1Projective {
        G1Projective(self.0 - rhs.0)
    }
}

impl Sub<&G1Affine> for &G1Projective {
    type Output = G1Projective;

    #[inline]
    fn sub(self, rhs: &G1Affine) -> G1Projective {
        G1Projective(self.0 - rhs.0)
    }
}

impl AddAssign<&G1Projective> for G1Projective {
    #[inline]
    fn add_assign(&mut self, rhs: &G1Projective) {
        *self = *self + rhs
    }
}

impl SubAssign<&G1Projective> for G1Projective {
    #[inline]
    fn sub_assign(&mut self, rhs: &G1Projective) {
        *self = *self - rhs;
    }
}

impl AddAssign<&G1Affine> for G1Projective {
    #[inline]
    fn add_assign(&mut self, rhs: &G1Affine) {
        *self = *self + rhs
    }
}

impl SubAssign<&G1Affine> for G1Projective {
    #[inline]
    fn sub_assign(&mut self, rhs: &G1Affine) {
        *self = *self - rhs;
    }
}

impl Mul<&Scalar> for &G1Projective {
    type Output = G1Projective;

    fn mul(self, scalar: &Scalar) -> Self::Output {
        G1Projective(self.0 * scalar.0)
    }
}

impl Mul<&Scalar> for &G1Affine {
    type Output = G1Projective;

    fn mul(self, scalar: &Scalar) -> Self::Output {
        G1Projective(self.0 * scalar.0)
    }
}

impl MulAssign<&Scalar> for G1Projective {
    #[inline]
    fn mul_assign(&mut self, rhs: &Scalar) {
        *self = *self * rhs;
    }
}

impl MulAssign<&Scalar> for G1Affine {
    #[inline]
    fn mul_assign(&mut self, rhs: &Scalar) {
        *self = (*self * rhs).into();
    }
}

impl_add_sub!(G1Projective);
impl_add_sub!(G1Projective, G1Affine);
impl_add_sub!(G1Affine, G1Projective, G1Projective);

impl_add_sub_assign!(G1Projective);
impl_add_sub_assign!(G1Projective, G1Affine);

impl_mul!(G1Projective, Scalar);
impl_mul!(G1Affine, Scalar, G1Projective);

impl_mul_assign!(G1Projective, Scalar);
impl_mul_assign!(G1Affine, Scalar);

impl From<G1Projective> for G1Affine {
    fn from(p: G1Projective) -> Self {
        Self(p.0.into())
    }
}

impl From<G1Affine> for G1Projective {
    fn from(p: G1Affine) -> Self {
        Self(p.0.into())
    }
}

impl From<blstrs::G1Projective> for G1Projective {
    fn from(p: blstrs::G1Projective) -> Self {
        Self(p)
    }
}

impl From<G1Affine> for blstrs::G1Affine {
    fn from(p: G1Affine) -> Self {
        p.0
    }
}

// This is implemented so that `G1Affine` can directly be used as `G1Prepared`.
impl From<&G1Affine> for G1Affine {
    fn from(p: &G1Affine) -> Self {
        *p
    }
}

// This is implemented so that `G1Affine` can directly be used as `G1Prepared`.
impl From<&G1Projective> for G1Affine {
    fn from(p: &G1Projective) -> Self {
        p.into()
    }
}

impl fmt::Display for G1Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_identity().into() {
            write!(f, "Infinity")
        } else {
            write!(f, "({}, {})", self.0.x(), self.0.y())
        }
    }
}

impl Zeroize for G1Affine {
    fn zeroize(&mut self) {
        self.0 = blstrs::G1Affine::identity();
    }
}

impl Distribution<G1Affine> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> G1Affine {
        G1Projective::from(blstrs::G1Projective::random(rng)).into()
    }
}

// The implementation is based on
// https://github.com/arkworks-rs/algebra/blob/118f8326c78d335f7c9f3c0b9077985a090b9d28/ec/src/models/short_weierstrass/affine.rs
impl AffineRepr for G1Affine {
    type Config = Config;
    type BaseField = <Self::Config as CurveConfig>::BaseField;
    type ScalarField = <Self::Config as CurveConfig>::ScalarField;
    type Group = G1Projective;

    fn xy(&self) -> Option<(Self::BaseField, Self::BaseField)> {
        if self.0.is_identity().into() {
            None
        } else {
            Some((Fp(self.0.x()), Fp(self.0.y())))
        }
    }

    #[inline]
    fn generator() -> Self {
        Self(blstrs::G1Affine::generator())
    }

    fn zero() -> Self {
        Self(blstrs::G1Affine::identity())
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        let mut b = [0u8; 32];
        b.copy_from_slice(bytes);
        Option::from(
            G1Projective(<blstrs::G1Projective as group::Group>::random(
                rand::rngs::StdRng::from_seed(b),
            ))
            .into_affine(),
        )
    }

    fn mul_bigint(&self, by: impl AsRef<[u64]>) -> Self::Group {
        let mut res = G1Projective::zero();
        for b in ark_ff::BitIteratorBE::without_leading_zeros(by) {
            res.double_in_place();
            if b {
                res += self
            }
        }

        res
    }

    /// Multiplies this element by the cofactor and output the
    /// resulting projective element.
    #[must_use]
    fn mul_by_cofactor_to_group(&self) -> Self::Group {
        self.mul_bigint(Self::Config::COFACTOR)
    }

    /// Performs cofactor clearing.
    /// The default method is simply to multiply by the cofactor.
    /// Some curves can implement a more efficient algorithm.
    fn clear_cofactor(&self) -> Self {
        self.mul_by_cofactor()
    }
}

impl CanonicalSerialize for G1Affine {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        match compress {
            ark_serialize::Compress::Yes => writer
                .write_all(&self.0.to_compressed())
                .map_err(|_| SerializationError::InvalidData)?,
            ark_serialize::Compress::No => writer
                .write_all(&self.0.to_uncompressed())
                .map_err(|_| SerializationError::InvalidData)?,
        };

        Ok(())
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        match compress {
            ark_serialize::Compress::Yes => COMPRESSED_SIZE,
            ark_serialize::Compress::No => UNCOMPRESSED_SIZE,
        }
    }
}

impl Valid for G1Affine {
    fn check(&self) -> Result<(), SerializationError> {
        // TODO vmx 2023-02-02: Check if blstrs `is_torsion_free()` is the same as arkworks
        // `is_in_correct_subgroup_assuming_on_curve()`.
        if self.0.is_on_curve().into() && self.0.is_torsion_free().into() {
            Ok(())
        } else {
            Err(SerializationError::InvalidData)
        }
    }
}

impl CanonicalDeserialize for G1Affine {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let blstrs_g1 = match compress {
            ark_serialize::Compress::Yes => {
                let mut bytes = [0u8; COMPRESSED_SIZE];
                reader
                    .read_exact(&mut bytes)
                    .ok()
                    .ok_or(SerializationError::InvalidData)?;
                blstrs::G1Affine::from_compressed_unchecked(&bytes).unwrap()
            }
            ark_serialize::Compress::No => {
                let mut bytes = [0u8; UNCOMPRESSED_SIZE];
                reader
                    .read_exact(&mut bytes)
                    .ok()
                    .ok_or(SerializationError::InvalidData)?;
                blstrs::G1Affine::from_uncompressed_unchecked(&bytes).unwrap()
            }
        };

        let g1 = G1Affine(blstrs_g1);

        if validate == ark_serialize::Validate::Yes {
            g1.check()?;
        }

        Ok(g1)
    }
}

// Implementations are based on
// https://github.com/arkworks-rs/algebra/blob/3448ccf72e8724486d0fa8e9a4de13e212f9077e/ec/src/models/short_weierstrass/group.rs
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct G1Projective(blstrs::G1Projective);

impl Deref for G1Projective {
    type Target = blstrs::G1Projective;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for G1Projective {
    fn default() -> Self {
        Self(blstrs::G1Projective::identity())
    }
}

// TODO check invariant
#[allow(unknown_lints, renamed_and_removed_lints)]
#[allow(clippy::derived_hash_with_manual_eq, clippy::derive_hash_xor_eq)]
impl Hash for G1Projective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_uncompressed()[..]);
    }
}

impl fmt::Display for G1Projective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", G1Affine::from(*self))
    }
}

impl Zeroize for G1Projective {
    fn zeroize(&mut self) {
        self.0 = blstrs::G1Projective::identity();
    }
}

impl Zero for G1Projective {
    #[inline]
    fn zero() -> Self {
        Self(blstrs::G1Projective::identity())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_identity().into()
    }
}

impl Distribution<G1Projective> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> G1Projective {
        G1Projective::from(blstrs::G1Projective::random(rng))
    }
}

impl Group for G1Projective {
    type ScalarField = <Config as CurveConfig>::ScalarField;

    #[inline]
    fn generator() -> Self {
        blstrs::G1Projective::generator().into()
    }

    /// Sets `self = 2 * self`. Note that Jacobian formulae are incomplete, and
    /// so doubling cannot be computed as `self + self`. Instead, this
    /// implementation uses the following specialized doubling formulae:
    /// * [`P::A` is zero](http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l)
    /// * [`P::A` is not zero](https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl)
    fn double_in_place(&mut self) -> &mut Self {
        *self = G1Projective::from(self.0.double());
        self
    }

    #[inline]
    fn mul_bigint(&self, _other: impl AsRef<[u64]>) -> Self {
        unimplemented!("mul_bigint")
        // Better be safe then sorry. The function below is likely correct. We'll know once we use
        // it.
        //// TODO vmx 2023-02-02: check if this code is actually doing the right thing. It was
        //// copied from `G1Affine::mul_bigint`.
        //let mut res = G1Projective::zero();
        //for b in ark_ff::BitIteratorBE::without_leading_zeros(other) {
        //    res.double_in_place();
        //    if b {
        //        res += self
        //    }
        //}
        //
        //res
    }
}

impl CurveGroup for G1Projective {
    type Config = Config;
    type BaseField = <Config as CurveConfig>::BaseField;
    type Affine = G1Affine;
    type FullGroup = G1Affine;

    #[inline]
    fn normalize_batch(projective: &[Self]) -> Vec<Self::Affine> {
        let blstrs_projective = projective.iter().map(|p| p.0).collect::<Vec<_>>();
        let mut blstrs_affine = vec![blstrs::G1Affine::identity(); projective.len()];
        assert!(blstrs_projective.len() == blstrs_affine.len());
        blstrs::G1Projective::batch_normalize(&blstrs_projective, &mut blstrs_affine[..]);
        blstrs_affine.into_iter().map(G1Affine).collect()
    }
}

impl CanonicalSerialize for G1Projective {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.into_affine().serialize_with_mode(writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        match compress {
            ark_serialize::Compress::Yes => COMPRESSED_SIZE,
            ark_serialize::Compress::No => UNCOMPRESSED_SIZE,
        }
    }
}

impl Valid for G1Projective {
    fn check(&self) -> Result<(), SerializationError> {
        self.into_affine().check()
    }

    fn batch_check<'a>(
        batch: impl Iterator<Item = &'a Self> + Send,
    ) -> Result<(), SerializationError>
    where
        Self: 'a,
    {
        let batch = batch.copied().collect::<Vec<_>>();
        let batch = Self::normalize_batch(&batch);
        G1Affine::batch_check(batch.iter())
    }
}

impl CanonicalDeserialize for G1Projective {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let affine = G1Affine::deserialize_with_mode(reader, compress, validate)?;
        Ok(affine.into())
    }
}

impl ScalarMul for G1Projective {
    type MulBase = G1Affine;
    const NEGATION_IS_CHEAP: bool = true;

    fn batch_convert_to_mul_base(bases: &[Self]) -> Vec<Self::MulBase> {
        Self::normalize_batch(bases)
    }
}

impl VariableBaseMSM for G1Projective {
    fn msm(bases: &[Self::MulBase], bigints: &[Self::ScalarField]) -> Result<Self, usize> {
        // NOTE vmx 2023-02-03: The bases are converted projective for the `blstrs` call.
        // Internally it then converts it to affine again. A possible optimization is to implement
        // a `blstrs::G1Affine::multi_exp` that takes the scalars directly in affine
        // representation.
        let blstrs_bases = bases
            .iter()
            .map(|base| G1Projective::from(*base).0)
            .collect::<Vec<_>>();
        let blstrs_bigints = bigints.iter().map(|bigint| bigint.0).collect::<Vec<_>>();
        Ok(G1Projective(blstrs::G1Projective::multi_exp(
            &blstrs_bases,
            &blstrs_bigints,
        )))
    }
}

impl<'a> core::iter::Sum<&'a G1Affine> for G1Projective {
    fn sum<I: Iterator<Item = &'a G1Affine>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

impl core::iter::Sum<G1Affine> for G1Projective {
    fn sum<I: Iterator<Item = G1Affine>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

// In Arkworks `impl_additive_ops_from_ref!()` implements this.
impl<'a> core::iter::Sum<&'a Self> for G1Projective {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

// In Arkworks `impl_additive_ops_from_ref!()` implements this.
impl core::iter::Sum<Self> for G1Projective {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

impl Neg for G1Projective {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.0 = -self.0;
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tests::group_test;

    #[test]
    fn g1() {
        group_test::<G1Projective>();
    }
}
