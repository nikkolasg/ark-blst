use core::{
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Neg, Sub, SubAssign},
};

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
use group::{prime::PrimeCurveAffine, Curve as _, Group as _};
use zeroize::Zeroize;

use crate::fp2::Fp2;
use crate::scalar::Scalar;

const COMPRESSED_SIZE: usize = 96;
const UNCOMPRESSED_SIZE: usize = 192;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Config;

impl CurveConfig for Config {
    type BaseField = Fp2;
    type ScalarField = Scalar;

    /// COFACTOR = (x^8 - 4 x^7 + 5 x^6) - (4 x^4 + 6 x^3 - 4 x^2 - 4 x + 13) //
    /// 9
    /// = 305502333931268344200999753193121504214466019254188142667664032982267604182971884026507427359259977847832272839041616661285803823378372096355777062779109
    const COFACTOR: &'static [u64] = &[
        0xcf1c38e31c7238e5,
        0x1616ec6e786f0c70,
        0x21537e293a6691ae,
        0xa628f1cb4d9e82ef,
        0xa68a205b2e5a7ddf,
        0xcd91de4547085aba,
        0x91d50792876a202,
        0x5d543a95414e7f1,
    ];

    /// COFACTOR_INV = COFACTOR^{-1} mod r
    /// 26652489039290660355457965112010883481355318854675681319708643586776743290055
    const COFACTOR_INV: Scalar = Scalar(blstrs::Scalar::from_raw_unchecked([
        6746407649509787816,
        1304054119431494378,
        2461312685643913071,
        5956596749362435284,
    ]));
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct G2Affine(blstrs::G2Affine);

impl Deref for G2Affine {
    type Target = blstrs::G2Affine;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO check invariant
#[allow(unknown_lints, renamed_and_removed_lints)]
#[allow(clippy::derived_hash_with_manual_eq, clippy::derive_hash_xor_eq)]
impl Hash for G2Affine {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_uncompressed()[..]);
    }
}

impl Neg for G2Affine {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.0 = -self.0;
        self
    }
}

impl Add<&G2Affine> for G2Affine {
    type Output = G2Projective;

    #[inline]
    fn add(self, rhs: &G2Affine) -> Self::Output {
        G2Projective(self.0 + blstrs::G2Projective::from(rhs.0))
    }
}

impl Add<G2Affine> for G2Affine {
    type Output = G2Projective;

    #[inline]
    fn add(self, rhs: G2Affine) -> Self::Output {
        G2Projective(self.0 + blstrs::G2Projective::from(&rhs.0))
    }
}

impl Add<&G2Projective> for &G2Projective {
    type Output = G2Projective;

    #[inline]
    fn add(self, rhs: &G2Projective) -> G2Projective {
        G2Projective(self.0 + rhs.0)
    }
}

impl Add<&G2Projective> for &G2Affine {
    type Output = G2Projective;

    #[inline]
    fn add(self, rhs: &G2Projective) -> G2Projective {
        G2Projective(self.0 + rhs.0)
    }
}

impl Add<&G2Affine> for &G2Projective {
    type Output = G2Projective;

    #[inline]
    fn add(self, rhs: &G2Affine) -> G2Projective {
        G2Projective(self.0 + rhs.0)
    }
}

impl Sub<&G2Projective> for &G2Projective {
    type Output = G2Projective;

    #[inline]
    fn sub(self, rhs: &G2Projective) -> G2Projective {
        G2Projective(self.0 - rhs.0)
    }
}

impl Sub<&G2Projective> for &G2Affine {
    type Output = G2Projective;

    #[inline]
    fn sub(self, rhs: &G2Projective) -> G2Projective {
        G2Projective(self.0 - rhs.0)
    }
}

impl Sub<&G2Affine> for &G2Projective {
    type Output = G2Projective;

    #[inline]
    fn sub(self, rhs: &G2Affine) -> G2Projective {
        G2Projective(self.0 - rhs.0)
    }
}

impl AddAssign<&G2Projective> for G2Projective {
    #[inline]
    fn add_assign(&mut self, rhs: &G2Projective) {
        *self = *self + rhs
    }
}

impl SubAssign<&G2Projective> for G2Projective {
    #[inline]
    fn sub_assign(&mut self, rhs: &G2Projective) {
        *self = *self - rhs;
    }
}

impl AddAssign<&G2Affine> for G2Projective {
    #[inline]
    fn add_assign(&mut self, rhs: &G2Affine) {
        *self = *self + rhs
    }
}

impl SubAssign<&G2Affine> for G2Projective {
    #[inline]
    fn sub_assign(&mut self, rhs: &G2Affine) {
        *self = *self - rhs;
    }
}

impl Mul<&Scalar> for &G2Projective {
    type Output = G2Projective;

    fn mul(self, scalar: &Scalar) -> Self::Output {
        G2Projective(self.0 * scalar.0)
    }
}

impl Mul<&Scalar> for &G2Affine {
    type Output = G2Projective;

    fn mul(self, scalar: &Scalar) -> Self::Output {
        G2Projective(self.0 * scalar.0)
    }
}

impl MulAssign<&Scalar> for G2Projective {
    #[inline]
    fn mul_assign(&mut self, rhs: &Scalar) {
        *self = *self * rhs;
    }
}

impl MulAssign<&Scalar> for G2Affine {
    #[inline]
    fn mul_assign(&mut self, rhs: &Scalar) {
        *self = (*self * rhs).into();
    }
}

impl_add_sub!(G2Projective);
impl_add_sub!(G2Projective, G2Affine);
impl_add_sub!(G2Affine, G2Projective, G2Projective);

impl_add_sub_assign!(G2Projective);
impl_add_sub_assign!(G2Projective, G2Affine);

impl_mul!(G2Projective, Scalar);
impl_mul!(G2Affine, Scalar, G2Projective);

impl_mul_assign!(G2Projective, Scalar);
impl_mul_assign!(G2Affine, Scalar);

impl From<G2Projective> for G2Affine {
    fn from(p: G2Projective) -> Self {
        Self(p.0.into())
    }
}

impl From<G2Affine> for G2Projective {
    fn from(p: G2Affine) -> Self {
        Self(p.0.into())
    }
}

impl From<blstrs::G2Projective> for G2Projective {
    fn from(p: blstrs::G2Projective) -> Self {
        Self(p)
    }
}

impl fmt::Display for G2Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_identity().into() {
            write!(f, "Infinity")
        } else {
            write!(f, "({}, {})", self.0.x(), self.0.y())
        }
    }
}

impl Zeroize for G2Affine {
    fn zeroize(&mut self) {
        self.0 = blstrs::G2Affine::identity();
    }
}

impl Distribution<G2Affine> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> G2Affine {
        G2Projective::from(blstrs::G2Projective::random(rng)).into()
    }
}

// The implementation is based on
// https://github.com/arkworks-rs/algebra/blob/118f8326c78d335f7c9f3c0b9077985a090b9d28/ec/src/models/short_weierstrass/affine.rs
impl AffineRepr for G2Affine {
    type Config = Config;
    type BaseField = <Self::Config as CurveConfig>::BaseField;
    type ScalarField = <Self::Config as CurveConfig>::ScalarField;
    type Group = G2Projective;

    fn xy(&self) -> Option<(Self::BaseField, Self::BaseField)> {
        if self.0.is_identity().into() {
            None
        } else {
            Some((Fp2(self.0.x()), Fp2(self.0.y())))
        }
    }

    #[inline]
    fn generator() -> Self {
        Self(blstrs::G2Affine::generator())
    }

    fn zero() -> Self {
        Self(blstrs::G2Affine::identity())
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Option::from(blstrs::G2Affine::from_uncompressed(bytes.try_into().unwrap()).map(Self))
    }

    fn mul_bigint(&self, by: impl AsRef<[u64]>) -> Self::Group {
        let mut res = G2Projective::zero();
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

impl CanonicalSerialize for G2Affine {
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

impl Valid for G2Affine {
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

impl CanonicalDeserialize for G2Affine {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let blstrs_g2 = match compress {
            ark_serialize::Compress::Yes => {
                let mut bytes = [0u8; COMPRESSED_SIZE];
                reader
                    .read_exact(&mut bytes)
                    .ok()
                    .ok_or(SerializationError::InvalidData)?;
                blstrs::G2Affine::from_compressed_unchecked(&bytes).unwrap()
            }
            ark_serialize::Compress::No => {
                let mut bytes = [0u8; UNCOMPRESSED_SIZE];
                reader
                    .read_exact(&mut bytes)
                    .ok()
                    .ok_or(SerializationError::InvalidData)?;
                blstrs::G2Affine::from_uncompressed_unchecked(&bytes).unwrap()
            }
        };

        let g2 = G2Affine(blstrs_g2);

        if validate == ark_serialize::Validate::Yes {
            g2.check()?;
        }

        Ok(g2)
    }
}

// Implementations are based on
// https://github.com/arkworks-rs/algebra/blob/3448ccf72e8724486d0fa8e9a4de13e212f9077e/ec/src/models/short_weierstrass/group.rs
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct G2Projective(blstrs::G2Projective);

impl Deref for G2Projective {
    type Target = blstrs::G2Projective;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for G2Projective {
    fn default() -> Self {
        Self(blstrs::G2Projective::identity())
    }
}

// TODO check invariant
#[allow(unknown_lints, renamed_and_removed_lints)]
#[allow(clippy::derived_hash_with_manual_eq, clippy::derive_hash_xor_eq)]
impl Hash for G2Projective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_uncompressed()[..]);
    }
}

impl fmt::Display for G2Projective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", G2Affine::from(*self))
    }
}

impl Zeroize for G2Projective {
    fn zeroize(&mut self) {
        self.0 = blstrs::G2Projective::identity();
    }
}

impl Zero for G2Projective {
    #[inline]
    fn zero() -> Self {
        Self(blstrs::G2Projective::identity())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_identity().into()
    }
}

impl Distribution<G2Projective> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> G2Projective {
        G2Projective::from(blstrs::G2Projective::random(rng))
    }
}

impl Group for G2Projective {
    type ScalarField = <Config as CurveConfig>::ScalarField;

    #[inline]
    fn generator() -> Self {
        blstrs::G2Projective::generator().into()
    }

    /// Sets `self = 2 * self`. Note that Jacobian formulae are incomplete, and
    /// so doubling cannot be computed as `self + self`. Instead, this
    /// implementation uses the following specialized doubling formulae:
    /// * [`P::A` is zero](http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l)
    /// * [`P::A` is not zero](https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl)
    fn double_in_place(&mut self) -> &mut Self {
        *self = G2Projective::from(self.0.double());
        self
    }

    #[inline]
    fn mul_bigint(&self, _other: impl AsRef<[u64]>) -> Self {
        unimplemented!("mul_bigint")
        // Better be safe then sorry. The function below is likely correct. We'll know once we use
        // it.
        //// TODO vmx 2023-02-02: check if this code is actually doing the right thing. It was
        //// copied from `G2Affine::mul_bigint`.
        //let mut res = G2Projective::zero();
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

impl CurveGroup for G2Projective {
    type Config = Config;
    type BaseField = <Config as CurveConfig>::BaseField;
    type Affine = G2Affine;
    type FullGroup = G2Affine;

    #[inline]
    fn normalize_batch(projective: &[Self]) -> Vec<Self::Affine> {
        let blstrs_projective = projective.iter().map(|p| p.0).collect::<Vec<_>>();
        let mut blstrs_affine = vec![blstrs::G2Affine::identity(); projective.len()];
        assert!(blstrs_projective.len() == blstrs_affine.len());
        blstrs::G2Projective::batch_normalize(&blstrs_projective, &mut blstrs_affine[..]);
        blstrs_affine.into_iter().map(G2Affine).collect()
    }
}

impl CanonicalSerialize for G2Projective {
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

impl Valid for G2Projective {
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
        G2Affine::batch_check(batch.iter())
    }
}

impl CanonicalDeserialize for G2Projective {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let affine = G2Affine::deserialize_with_mode(reader, compress, validate)?;
        Ok(affine.into())
    }
}

impl ScalarMul for G2Projective {
    type MulBase = G2Affine;
    const NEGATION_IS_CHEAP: bool = true;

    fn batch_convert_to_mul_base(bases: &[Self]) -> Vec<Self::MulBase> {
        Self::normalize_batch(bases)
    }
}

impl VariableBaseMSM for G2Projective {
    fn msm(bases: &[Self::MulBase], bigints: &[Self::ScalarField]) -> Result<Self, usize> {
        // NOTE vmx 2023-02-03: The bases are converted projective for the `blstrs` call.
        // Internally it then converts it to affine again. A possible optimization is to implement
        // a `blstrs::G2Affine::multi_exp` that takes the scalars directly in affine
        // representation.
        let blstrs_bases = bases
            .iter()
            .map(|base| G2Projective::from(*base).0)
            .collect::<Vec<_>>();
        let blstrs_bigints = bigints.iter().map(|bigint| bigint.0).collect::<Vec<_>>();
        Ok(G2Projective(blstrs::G2Projective::multi_exp(
            &blstrs_bases,
            &blstrs_bigints,
        )))
    }
}

impl<'a> core::iter::Sum<&'a G2Affine> for G2Projective {
    fn sum<I: Iterator<Item = &'a G2Affine>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

impl core::iter::Sum<G2Affine> for G2Projective {
    fn sum<I: Iterator<Item = G2Affine>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

// In Arkworks `impl_additive_ops_from_ref!()` implements this.
impl<'a> core::iter::Sum<&'a Self> for G2Projective {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

// In Arkworks `impl_additive_ops_from_ref!()` implements this.
impl core::iter::Sum<Self> for G2Projective {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, x| sum + x)
    }
}

impl Neg for G2Projective {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.0 = -self.0;
        self
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct G2Prepared(blstrs::G2Prepared);

impl G2Prepared {
    pub fn is_identity(&self) -> bool {
        self.0.is_identity().into()
    }
}

impl Default for G2Prepared {
    fn default() -> Self {
        Self::from(G2Affine::generator())
    }
}

impl From<G2Affine> for G2Prepared {
    fn from(q: G2Affine) -> Self {
        G2Prepared(blstrs::G2Prepared::from(q.0))
    }
}

impl From<G2Projective> for G2Prepared {
    fn from(q: G2Projective) -> Self {
        q.into_affine().into()
    }
}

impl From<&G2Affine> for G2Prepared {
    fn from(other: &G2Affine) -> Self {
        (*other).into()
    }
}

impl From<&G2Projective> for G2Prepared {
    fn from(q: &G2Projective) -> Self {
        q.into_affine().into()
    }
}

impl From<G2Prepared> for blstrs::G2Prepared {
    fn from(p: G2Prepared) -> Self {
        p.0
    }
}

impl CanonicalSerialize for G2Prepared {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut _writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        todo!("canonical_serialize")
    }

    #[inline]
    fn serialized_size(&self, _compress: Compress) -> usize {
        todo!("canonical_serialize_size")
    }
}

impl Valid for G2Prepared {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for G2Prepared {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        todo!("canonical_deserialize")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tests::group_test;

    #[test]
    fn g2() {
        group_test::<G2Projective>();
    }
}
