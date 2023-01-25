use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use std::io::Empty;
use std::ops::{Div, DivAssign};
use std::str::FromStr;

use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, EmptyFlags, Flags, SerializationError,
    Valid, Validate,
};
use ff::Field;
use num_traits::{One, Zero};
use std::iter;
use std::{hash::Hasher, ops::Deref};
use zeroize::Zeroize;

// Little-endian non-Montgomery form.
#[allow(dead_code)]
const MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];
const MODULUS_BIGINT: ark_ff::BigInteger384 = ark_ff::BigInt::<6>(MODULUS);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Fp(blstrs::Fp);

impl Deref for Fp {
    type Target = blstrs::Fp;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl AddAssign<&Fp> for Fp {
    #[inline]
    fn add_assign(&mut self, rhs: &Fp) {
        self.0.add_assign(rhs.0)
    }
}

impl AddAssign<Fp> for Fp {
    #[inline]
    fn add_assign(&mut self, rhs: Fp) {
        self.0.add_assign(rhs.0)
    }
}

//impl<'a> AddAssign<&'a mut Fp> for Fp {
//    fn add_assign(&mut self, rhs: &'a mut Fp) {
//        self.add_assign(rhs);
//    }
//}

impl Add<Fp> for Fp {
    type Output = Fp;

    #[inline]
    fn add(mut self, rhs: Fp) -> Fp {
        self += &rhs;
        self
    }
}

impl Add<&Fp> for &Fp {
    type Output = Fp;

    #[inline]
    fn add(self, rhs: &Fp) -> Fp {
        let mut out = *self;
        out += rhs;
        out
    }
}

impl Add<&Fp> for Fp {
    type Output = Fp;
    #[inline]
    fn add(self, rhs: &Fp) -> Fp {
        Fp(self.0.add(rhs.0))
    }
}
impl Neg for &Fp {
    type Output = Fp;

    #[inline]
    fn neg(self) -> Fp {
        Fp(self.0.neg())
    }
}

impl Neg for Fp {
    type Output = Fp;

    #[inline]
    fn neg(self) -> Fp {
        Fp(self.0.neg())
    }
}

impl SubAssign<&Fp> for Fp {
    #[inline]
    fn sub_assign(&mut self, rhs: &Fp) {
        self.0.sub_assign(rhs.0);
    }
}

//impl<'a> SubAssign<&'a mut Fp> for Fp {
//    fn sub_assign(&mut self, rhs: &'a mut Fp) {
//        self.sub_assign(rhs);
//    }
//}

impl SubAssign<Fp> for Fp {
    fn sub_assign(&mut self, rhs: Fp) {
        self.0.sub_assign(rhs.0)
    }
}

impl Sub<&Fp> for &Fp {
    type Output = Fp;

    #[inline]
    fn sub(self, rhs: &Fp) -> Fp {
        let mut out = *self;
        out.0 -= rhs.0;
        out
    }
}

//impl<'a> Sub<&'a mut Fp> for Fp {
//    type Output = Fp;
//
//    fn sub(self, rhs: &'a mut Fp) -> Self::Output {
//        self.sub(rhs)
//    }
//}
//
//impl<'a> Sub<&'a Fp> for Fp {
//    type Output = Fp;
//
//    fn sub(self, rhs: &'a Fp) -> Self::Output {
//        self.sub(rhs)
//    }
//}
impl MulAssign<&Fp> for Fp {
    #[inline]
    fn mul_assign(&mut self, rhs: &Fp) {
        self.0.mul_assign(rhs.0);
    }
}

impl MulAssign<Fp> for Fp {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0);
    }
}

//impl<'a> MulAssign<&'a mut Fp> for Fp {
//    fn mul_assign(&mut self, rhs: &'a mut Fp) {
//        self.mul_assign(rhs);
//    }
//}

impl Mul<&Fp> for Fp {
    type Output = Fp;

    #[inline]
    fn mul(mut self, rhs: &Fp) -> Fp {
        self.0.mul_assign(rhs.0);
        self
    }
}

impl Mul<Fp> for Fp {
    type Output = Fp;
    #[inline]
    fn mul(mut self, rhs: Fp) -> Fp {
        self.0.mul_assign(rhs.0);
        self
    }
}

impl Mul<&Fp> for &Fp {
    type Output = Fp;

    #[inline]
    fn mul(self, rhs: &Fp) -> Fp {
        let mut out = *self;
        out *= rhs;
        out
    }
}

impl One for Fp {
    fn one() -> Self {
        Fp(blstrs::Fp::one())
    }
}

impl Zero for Fp {
    fn zero() -> Self {
        Fp(blstrs::Fp::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }
}

impl Zeroize for Fp {
    fn zeroize(&mut self) {
        self.0 = blstrs::Fp::from(0);
    }
}
// TODO check invariant
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for Fp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_bytes_le()[..]);
    }
}

impl Valid for Fp {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

//impl ark_serialize::CanonicalDeserialize for Fp {
//    fn deserialize_with_mode<R: ark_serialize::Read>(
//        mut reader: R,
//        _compress: Compress,
//        _validate: Validate,
//    ) -> Result<Self, SerializationError> {
//        let mut buff = [0u8; 48];
//        reader
//            .read(&mut buff[..])
//            .map_err(|_| SerializationError::InvalidData)?;
//        Option::from(blstrs::Fp::from_bytes_le(&buff).map(|f| Fp(f)))
//            .ok_or(SerializationError::InvalidData)
//    }
//}

//impl ark_serialize::CanonicalDeserializeWithFlags for Fp {
//    fn deserialize_with_flags<R: ark_serialize::Read, F: Flags>(
//        reader: R,
//    ) -> Result<(Self, F), SerializationError> {
//        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
//            .map(|s| (s, F::from_u8(0).unwrap()))
//    }
//}

impl ark_serialize::CanonicalSerialize for Fp {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer
            .write(&self.0.to_bytes_le()[..])
            .map(|_| ())
            .map_err(|_| SerializationError::InvalidData)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        48
    }
}

//impl ark_serialize::CanonicalSerializeWithFlags for Fp {
//    fn serialize_with_flags<W: ark_serialize::Write, F: Flags>(
//        &self,
//        writer: W,
//        _flags: F,
//    ) -> Result<(), SerializationError> {
//        Self::serialize_with_mode(&self, writer, Compress::No)
//    }
//
//    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
//        Self::serialized_size(&self, Compress::No)
//    }
//}

//impl From<bool> for Fp {
//    fn from(value: bool) -> Self {
//        match value {
//            true => Fp::from(1u64),
//            false => Fp::from(0u64),
//        }
//    }
//}

//macro_rules! impl_from {
//    ($t:ty) => {
//        impl From<$t> for Fp {
//            fn from(value: $t) -> Self {
//                Fp(blstrs::Fp::from(value))
//            }
//        }
//    };
//}
//
//impl_from!(u8);
//impl_from!(u16);
//impl_from!(u32);
//impl_from!(u64);
//impl_from!(u128);
//
//impl<'a> core::iter::Product<&'a Fp> for Fp {
//    fn product<I: Iterator<Item = &'a Fp>>(iter: I) -> Self {
//        iter.fold(Fp::one(), |acc, x| acc * x)
//    }
//}
//
//impl core::iter::Product<Fp> for Fp {
//    fn product<I: Iterator<Item = Fp>>(iter: I) -> Self {
//        iter.fold(Fp::one(), |acc, x| acc * x)
//    }
//}
//
//impl<'a> core::iter::Sum<&'a Fp> for Fp {
//    fn sum<I: Iterator<Item = &'a Fp>>(iter: I) -> Self {
//        iter.fold(Fp::zero(), |acc, x| acc + x)
//    }
//}
//
//impl core::iter::Sum<Fp> for Fp {
//    fn sum<I: Iterator<Item = Fp>>(iter: I) -> Self {
//        iter.fold(Fp::zero(), |acc, x| acc + x)
//    }
//}
//
//impl Div<Fp> for Fp {
//    type Output = Fp;
//
//    fn div(self, rhs: Fp) -> Self::Output {
//        rhs.invert().unwrap() * self
//    }
//}
//
//impl<'a> Div<&'a mut Fp> for Fp {
//    type Output = Fp;
//
//    fn div(self, rhs: &'a mut Fp) -> Self::Output {
//        self.div_assign(rhs);
//    }
//}
//impl<'a> Div<&'a Fp> for Fp {
//    type Output = Fp;
//
//    fn div(self, rhs: &'a Fp) -> Self::Output {
//        self.div_assign(rhs);
//    }
//}
//impl DivAssign for Fp {
//    fn div_assign(&mut self, rhs: Self) {
//        *self *= rhs.inverse().unwrap();
//    }
//}
//
//impl<'a> DivAssign<&'a Fp> for Fp {
//    fn div_assign(&mut self, rhs: &'a Fp) {
//        *self *= rhs.inverse().unwrap();
//    }
//}
//
//impl<'a> DivAssign<&'a mut Fp> for Fp {
//    fn div_assign(&mut self, rhs: &'a mut Fp) {
//        *self *= rhs.inverse().unwrap();
//    }
//}
//
//impl<'a> Mul<&'a mut Fp> for Fp {
//    type Output = Fp;
//
//    fn mul(self, rhs: &'a mut Fp) -> Self::Output {
//        self.mul(rhs)
//    }
//}
//
//impl<'a> Add<&'a mut Fp> for Fp {
//    type Output = Fp;
//
//    fn add(self, rhs: &'a mut Fp) -> Self::Output {
//        self.add(rhs)
//    }
//}
//
//impl Sub<Fp> for Fp {
//    type Output = Fp;
//
//    fn sub(self, rhs: Fp) -> Self::Output {
//        Fp(self.0.sub(rhs.0))
//    }
//}
//
////impl rand::distribution::Standard for Fp
////where
////    rand::distributions::Standard: rand::distributions::Distribution<u64>,
////{
////    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self {
////        Fp(blstrs::Fp::random(rng))
////    }
////}
//
//impl ark_ff::UniformRand for Fp {
//    //fn rand<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(rng: &mut R) -> Self {
//    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
//        Fp(blstrs::Fp::random(rng))
//    }
//}
//impl From<num_bigint::BigUint> for Fp {
//    fn from(value: num_bigint::BigUint) -> Self {
//        Fp(blstrs::Fp::from(value))
//    }
//}
//
//impl From<BigInt<6>> for Fp {
//    fn from(value: BigInt<6>) -> Self {
//        Fp(blstrs::Fp::from(value))
//    }
//}
//
//impl FromStr for Fp {
//    type Err = SerializationError;
//
//    fn from_str(s: &str) -> Result<Self, Self::Err> {
//        Ok(Fp(
//            blstrs::Fp::from_str_vartime(s).map_err(|_| SerializationError::ParseError)?
//        ))
//    }
//}
//
////impl ark_ff::ToBytes for Fp {
////    #[inline]
////    fn write<W: ark_std::io::Write>(&self, mut writer: W) -> ark_std::io::Result<()> {
////        let mut bytes = [0u8; 48];
////        self.0.to_bytes(&mut bytes);
////        writer.write_all(&bytes)
////    }
////}
////
////impl ark_ff::FromBytes for Fp {
////    #[inline]
////    fn read<R: ark_std::io::Read>(mut reader: R) -> ark_std::io::Result<Self> {
////        let mut bytes = [0u8; 48];
////        reader.read_exact(&mut bytes)?;
////        Ok(Fp(blstrs::Fp::from_bytes(&bytes)))
////    }
////}
//
//impl From<ark_bls12_381::Fq> for Fp {
//    fn from(value: ark_bls12_381::Fq) -> Self {
//        Fp::from(value.into_bigint())
//    }
//}
//
//impl ark_ff::FftField for Fp {
//    const GENERATOR: Self = Fp::one();
//
//    const TWO_ADICITY: u32 = <ark_bls12_381::Fq as ark_ff::FftField>::TWO_ADICITY;
//
//    const TWO_ADIC_ROOT_OF_UNITY: Self =
//        Fp::from(<ark_bls12_381::Fq as ark_ff::FftField>::TWO_ADIC_ROOT_OF_UNITY);
//}
//
//impl Into<num_bigint::BigUint> for Fp {
//    fn into(self) -> num_bigint::BigUint {
//        self.0.into()
//    }
//}
//
//impl Into<<Fp as PrimeField>::BigInt> for Fp {
//    fn into(self) -> <Fp as PrimeField>::BigInt {
//        self.0.into()
//    }
//}
//
//impl ark_ff::PrimeField for Fp {
//    type BigInt = ark_ff::BigInteger384;
//
//    const MODULUS: Self::BigInt = MODULUS_BIGINT;
//
//    // TODO : currently using bls12-381 for quick and dirty - best to copy the constants here
//    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt =
//        <ark_bls12_381::Fq as ark_ff::PrimeField>::MODULUS_MINUS_ONE_DIV_TWO;
//
//    const MODULUS_BIT_SIZE: u32 = <ark_bls12_381::Fq as ark_ff::PrimeField>::MODULUS_BIT_SIZE;
//
//    const TRACE: Self::BigInt = <ark_bls12_381::Fq as ark_ff::PrimeField>::TRACE;
//
//    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt =
//        <ark_bls12_381::Fq as ark_ff::PrimeField>::TRACE_MINUS_ONE_DIV_TWO;
//
//    fn from_bigint(repr: Self::BigInt) -> Option<Self> {
//        Some(Fp(blstrs::Fp::from_raw_unchecked(repr.0)))
//    }
//
//    fn into_bigint(self) -> Self::BigInt {
//        self.0.into_raw()
//    }
//}
//
//impl ark_ff::Field for Fp {
//    type BasePrimeField = Fp;
//
//    type BasePrimeFieldIter = iter::Once<Self::BasePrimeField>;
//
//    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;
//
//    const ZERO: Self = Fp::zero();
//
//    const ONE: Self = Fp::one();
//
//    fn extension_degree() -> u64 {
//        1
//    }
//
//    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
//        iter::once(*self)
//    }
//
//    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
//        if elems.len() != (Self::extension_degree() as usize) {
//            return None;
//        }
//        Some(elems[0])
//    }
//
//    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
//        elem
//    }
//
//    #[inline]
//    fn double(&self) -> Self {
//        Fp(self.0.double())
//    }
//
//    fn double_in_place(&mut self) -> &mut Self {
//        self.0 = self.0.double()
//    }
//
//    fn neg_in_place(&mut self) -> &mut Self {
//        self.0 = self.0.neg();
//    }
//
//    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
//        let blst_buffer = [0u8; 48];
//        blst_buffer[..].copy_from_slice(bytes);
//        blstrs::Fp::from_bytes_le(&blst_buffer);
//    }
//
//    fn legendre(&self) -> ark_ff::LegendreSymbol {
//        todo!()
//    }
//
//    fn square(&self) -> Self {
//        Fp(self.0.square())
//    }
//
//    fn square_in_place(&mut self) -> &mut Self {
//        self.0 = self.0.square();
//    }
//
//    fn inverse(&self) -> Option<Self> {
//        self.0.invert()
//    }
//
//    fn inverse_in_place(&mut self) -> Option<&mut Self> {
//        self.0 = self.0.invert();
//    }
//
//    fn frobenius_map_in_place(&mut self, power: usize) {
//        todo!()
//    }
//
//    fn characteristic() -> &'static [u64] {
//        Self::BasePrimeField::characteristic()
//    }
//
//    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
//        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
//    }
//
//    fn sqrt(&self) -> Option<Self> {
//        match Self::SQRT_PRECOMP {
//            Some(tv) => tv.sqrt(self),
//            None => self.0.sqrt(),
//        }
//    }
//
//    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
//        (*self).sqrt().map(|sqrt| {
//            *self = sqrt;
//            self
//        })
//    }
//
//    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
//        let mut sum = Self::zero();
//        for i in 0..a.len() {
//            sum += a[i] * b[i];
//        }
//        sum
//    }
//
//    fn frobenius_map(&self, power: usize) -> Self {
//        let mut this = *self;
//        this.frobenius_map_in_place(power);
//        this
//    }
//
//    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
//        let mut res = Self::one();
//
//        for i in ark_ff::BitIteratorBE::without_leading_zeros(exp) {
//            res.square_in_place();
//
//            if i {
//                res *= self;
//            }
//        }
//        res
//    }
//
//    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
//        let mut res = Self::one();
//        for (pow, bit) in ark_ff::BitIteratorLE::without_trailing_zeros(exp).enumerate() {
//            if bit {
//                res *= powers_of_2.get(pow)?;
//            }
//        }
//        Some(res)
//    }
//}

#[cfg(test)]
mod tests {
    use ark_ff::BigInteger;
    use ark_ff::PrimeField;
    use blst::*;

    fn print_slice_hex(slice: &[u64]) -> String {
        [
            "[",
            &slice
                .iter()
                .map(|i| format!("{i:#x}"))
                .collect::<Vec<String>>()
                .join(", "),
            "]",
        ]
        .concat()
    }
    fn print_info<F: PrimeField>() {
        println!("MODULUS: {}", print_slice_hex(F::MODULUS.as_ref()));
        println!(
            "MODULUS_MINUS_ONE_DIV_TWO: {}",
            print_slice_hex(F::MODULUS_MINUS_ONE_DIV_TWO.as_ref())
        );
    }

    #[test]
    fn print_info_fq() {
        print_info::<ark_bls12_381::Fq>();
    }

    #[test]
    fn test_fq_modulus() {
        const printed_modulus: [u64; 6] = [
            0xb9feffffffffaaab,
            0x1eabfffeb153ffff,
            0x6730d2a0f6b0f624,
            0x64774b84f38512bf,
            0x4b1ba7b6434bacd7,
            0x1a0111ea397fe69a,
        ];
        const modo: super::Fp = super::Fp(blstrs::Fp(blst_fp { l: printed_modulus }));
        println!("{:?}", modo);
        const slice_mod_minus_one_div_two: [u64; 6] = [
            0xdcff7fffffffd555,
            0xf55ffff58a9ffff,
            0xb39869507b587b12,
            0xb23ba5c279c2895f,
            0x258dd3db21a5d66b,
            0xd0088f51cbff34d,
        ];
        const modmodt: super::Fp = super::Fp(blstrs::Fp(blst_fp {
            l: slice_mod_minus_one_div_two,
        }));
        println!("0x{}", hex::encode(modmodt.to_bytes_le()));
        println!(
            "0x{}",
            hex::encode(ark_bls12_381::Fq::MODULUS_MINUS_ONE_DIV_TWO.to_bytes_le())
        );
    }

    //    #[test]
    //    fn fp_playground() {
    //        let fp1: Fp = Fp::one();
    //        let fp2: Fp = Fp::one();
    //        let f2: Fp = fp1 + fp2;
    //        let expf2: Fp = Fp(blstrs::Fp::from(2u64));
    //        assert_eq!(f2, expf2);
    //
    //        //let f41: Fp = f2.double();
    //        let f42: Fp = f2 * f2;
    //        let expf4: Fp = Fp(blstrs::Fp::from(4u64));
    //        assert_eq!(f42, expf4);
    //        assert_eq!(f42, f42);
    //
    //        let mut buff = Vec::new();
    //        expf4.serialize_compressed(&mut buff).unwrap();
    //        let readf4 = Fp::deserialize_compressed(&buff[..]).unwrap();
    //        assert_eq!(expf4, readf4);
    //    }
}
