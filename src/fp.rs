use crate::memory;
use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::BigInteger;
use std::ops::{Div, DivAssign};
use std::str::FromStr;

use ark_ff::{BigInt, PrimeField};
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

impl<'a> AddAssign<&'a mut Fp> for Fp {
    fn add_assign(&mut self, rhs: &'a mut Fp) {
        self.0.add_assign(rhs.0);
    }
}

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

impl<'a> SubAssign<&'a mut Fp> for Fp {
    fn sub_assign(&mut self, rhs: &'a mut Fp) {
        self.0.sub_assign(rhs.0);
    }
}

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

impl<'a> MulAssign<&'a mut Fp> for Fp {
    fn mul_assign(&mut self, rhs: &'a mut Fp) {
        self.0.mul_assign(rhs.0);
    }
}

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

impl<'a> Mul<&'a mut Fp> for Fp {
    type Output = Fp;

    fn mul(self, rhs: &'a mut Fp) -> Self::Output {
        Fp(self.0.mul(rhs.0))
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
#[allow(clippy::derived_hash_with_manual_eq)]
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

impl ark_serialize::CanonicalDeserialize for Fp {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut buff = [0u8; 48];
        reader
            .read(&mut buff[..])
            .map_err(|_| SerializationError::InvalidData)?;
        Option::from(blstrs::Fp::from_bytes_le(&buff).map(|f| Fp(f)))
            .ok_or(SerializationError::InvalidData)
    }
}

impl ark_serialize::CanonicalDeserializeWithFlags for Fp {
    fn deserialize_with_flags<R: ark_serialize::Read, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
            .map(|s| (s, F::from_u8(0).unwrap()))
    }
}

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

impl ark_serialize::CanonicalSerializeWithFlags for Fp {
    fn serialize_with_flags<W: ark_serialize::Write, F: Flags>(
        &self,
        writer: W,
        _flags: F,
    ) -> Result<(), SerializationError> {
        Self::serialize_with_mode(&self, writer, Compress::No)
    }

    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        Self::serialized_size(&self, Compress::No)
    }
}

impl From<bool> for Fp {
    fn from(value: bool) -> Self {
        match value {
            true => Fp::from(1u64),
            false => Fp::from(0u64),
        }
    }
}

macro_rules! impl_from {
    ($t:ty) => {
        impl From<$t> for Fp {
            fn from(value: $t) -> Self {
                Fp(blstrs::Fp::from(value as u64))
            }
        }
    };
}

impl_from!(u8);
impl_from!(u16);
impl_from!(u32);
impl_from!(u64);
impl_from!(u128);

impl<'a> core::iter::Product<&'a Fp> for Fp {
    fn product<I: Iterator<Item = &'a Fp>>(iter: I) -> Self {
        iter.fold(Fp::one(), |acc, x| acc * x)
    }
}

impl core::iter::Product<Fp> for Fp {
    fn product<I: Iterator<Item = Fp>>(iter: I) -> Self {
        iter.fold(Fp::one(), |acc, x| acc * x)
    }
}

impl<'a> core::iter::Sum<&'a Fp> for Fp {
    fn sum<I: Iterator<Item = &'a Fp>>(iter: I) -> Self {
        iter.fold(Fp::zero(), |acc, x| acc + x)
    }
}

impl core::iter::Sum<Fp> for Fp {
    fn sum<I: Iterator<Item = Fp>>(iter: I) -> Self {
        iter.fold(Fp::zero(), |acc, x| acc + x)
    }
}

impl Div<Fp> for Fp {
    type Output = Fp;

    fn div(self, rhs: Fp) -> Self::Output {
        self * Fp(rhs.0.invert().unwrap())
    }
}

impl<'a> Div<&'a mut Fp> for Fp {
    type Output = Fp;

    fn div(self, rhs: &'a mut Fp) -> Self::Output {
        let mut c = self.clone();
        c.div_assign(rhs);
        c
    }
}
impl<'a> Div<&'a Fp> for Fp {
    type Output = Fp;

    fn div(self, rhs: &'a Fp) -> Self::Output {
        let mut c = self.clone();
        c.div_assign(rhs);
        c
    }
}
impl DivAssign for Fp {
    fn div_assign(&mut self, rhs: Self) {
        *self *= Fp(rhs.0.invert().unwrap());
    }
}

impl<'a> DivAssign<&'a Fp> for Fp {
    fn div_assign(&mut self, rhs: &'a Fp) {
        *self *= Fp(rhs.0.invert().unwrap());
    }
}

impl<'a> DivAssign<&'a mut Fp> for Fp {
    fn div_assign(&mut self, rhs: &'a mut Fp) {
        *self *= Fp(rhs.0.invert().unwrap());
    }
}

impl<'a> Add<&'a mut Fp> for Fp {
    type Output = Fp;

    fn add(self, rhs: &'a mut Fp) -> Self::Output {
        Fp(self.0.add(rhs.0))
    }
}

impl Sub<Fp> for Fp {
    type Output = Fp;

    fn sub(self, rhs: Fp) -> Self::Output {
        Fp(self.0.sub(rhs.0))
    }
}

impl<'a> Sub<&'a Fp> for Fp {
    type Output = Fp;

    fn sub(self, rhs: &'a Fp) -> Self::Output {
        Fp(self.0.sub(rhs.0))
    }
}

impl<'a> Sub<&'a mut Fp> for Fp {
    type Output = Fp;

    fn sub(self, rhs: &'a mut Fp) -> Self::Output {
        Fp(self.0.sub(rhs.0))
    }
}

//impl rand::distribution::Standard for Fp
//where
//    rand::distributions::Standard: rand::distributions::Distribution<u64>,
//{
//    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self {
//        Fp(blstrs::Fp::random(rng))
//    }
//}

impl ark_ff::UniformRand for Fp {
    //fn rand<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(rng: &mut R) -> Self {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Fp(blstrs::Fp::random(rng))
    }
}
impl From<num_bigint::BigUint> for Fp {
    fn from(value: num_bigint::BigUint) -> Self {
        Fp(
            blstrs::Fp::from_bytes_le(memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl From<BigInt<6>> for Fp {
    fn from(value: BigInt<6>) -> Self {
        Fp(
            blstrs::Fp::from_bytes_le(memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl FromStr for Fp {
    type Err = SerializationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ark_fq = ark_bls12_381::Fq::from_str(s).map_err(|_| SerializationError::InvalidData)?;
        Ok(Fp::from(ark_fq.into_bigint()))
    }
}

impl From<ark_bls12_381::Fq> for Fp {
    fn from(value: ark_bls12_381::Fq) -> Self {
        Fp::from(value.into_bigint())
    }
}

impl Into<num_bigint::BigUint> for Fp {
    fn into(self) -> num_bigint::BigUint {
        let slice = self.0.to_bytes_le();
        let s = memory::constant_size_to_slice(&slice);
        num_bigint::BigUint::from_bytes_le(s)
    }
}

impl Into<<Fp as PrimeField>::BigInt> for Fp {
    fn into(self) -> <Fp as PrimeField>::BigInt {
        let bg: num_bigint::BigUint = self.into();
        BigInt::<6>::try_from(bg).unwrap()
    }
}

impl ark_ff::FftField for Fp {
    const GENERATOR: Self = Fp(blstrs::fp::R);

    const TWO_ADICITY: u32 = <ark_bls12_381::Fq as ark_ff::FftField>::TWO_ADICITY;

    //        Fp::from(<ark_bls12_381::Fq as ark_ff::FftField>::TWO_ADIC_ROOT_OF_UNITY);
    const TWO_ADIC_ROOT_OF_UNITY: Self = Fp(blstrs::Fp(blst::blst_fp {
        l: [
            0xb9feffffffffaaaa,
            0x1eabfffeb153ffff,
            0x6730d2a0f6b0f624,
            0x64774b84f38512bf,
            0x4b1ba7b6434bacd7,
            0x1a0111ea397fe69a,
        ],
    }));
}

impl ark_ff::PrimeField for Fp {
    type BigInt = ark_ff::BigInteger384;

    const MODULUS: Self::BigInt = MODULUS_BIGINT;

    // TODO : currently using bls12-381 for quick and dirty - best to copy the constants here
    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt =
        <ark_bls12_381::Fq as ark_ff::PrimeField>::MODULUS_MINUS_ONE_DIV_TWO;
    //Fp(blstrs::Fp(blst_fp { l: [0xa1fafffffffe5557, 0x995bfff976a3fffe, 0x3f41d24d174ceb4, 0xf6547998c1995dbd, 0x778a468f507a6034, 0x20559931f7f8103] })))

    const MODULUS_BIT_SIZE: u32 = <ark_bls12_381::Fq as ark_ff::PrimeField>::MODULUS_BIT_SIZE;

    const TRACE: Self::BigInt = <ark_bls12_381::Fq as ark_ff::PrimeField>::TRACE;

    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt =
        <ark_bls12_381::Fq as ark_ff::PrimeField>::TRACE_MINUS_ONE_DIV_TWO;

    fn from_bigint(repr: Self::BigInt) -> Option<Self> {
        Some(Fp(blstrs::Fp::from_bytes_le(
            memory::slice_to_constant_size(repr.to_bytes_le().as_slice()),
        )
        .unwrap()))
    }

    fn into_bigint(self) -> Self::BigInt {
        self.into()
    }
}

impl ark_ff::Field for Fp {
    type BasePrimeField = Fp;

    type BasePrimeFieldIter = iter::Once<Self::BasePrimeField>;

    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;

    const ZERO: Self = Fp(blstrs::fp::ZERO);

    const ONE: Self = Fp(blstrs::fp::R);

    fn extension_degree() -> u64 {
        1
    }

    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        iter::once(*self)
    }

    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        if elems.len() != (Self::extension_degree() as usize) {
            return None;
        }
        Some(elems[0])
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        elem
    }

    #[inline]
    fn double(&self) -> Self {
        Fp(self.0.double())
    }

    fn double_in_place(&mut self) -> &mut Self {
        self.0 = self.0.double();
        self
    }

    fn neg_in_place(&mut self) -> &mut Self {
        self.0 = self.0.neg();
        self
    }

    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        let mut blst_buffer = [0u8; 48];
        blst_buffer[..].copy_from_slice(bytes);
        blstrs::Fp::from_bytes_le(&blst_buffer)
            .map(|fp| (Fp(fp), F::from_u8(0).unwrap()))
            .into()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        Fp(self.0.square())
    }

    fn square_in_place(&mut self) -> &mut Self {
        self.0 = self.0.square();
        self
    }

    fn inverse(&self) -> Option<Self> {
        self.0.invert().map(|f| Fp(f)).into()
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.0
            .invert()
            .map(|f| {
                self.0 = f;
                self
            })
            .into()
    }

    // nothing on base prime field
    fn frobenius_map_in_place(&mut self, _power: usize) {}

    fn characteristic() -> &'static [u64] {
        blstrs::fp::MODULUS[..].as_ref()
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
    }

    fn sqrt(&self) -> Option<Self> {
        match Self::SQRT_PRECOMP {
            Some(tv) => tv.sqrt(self),
            None => self.0.sqrt().map(|f| Fp(f)).into(),
        }
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        (*self).sqrt().map(|sqrt| {
            *self = sqrt;
            self
        })
    }

    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        let mut sum = Self::zero();
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    fn frobenius_map(&self, power: usize) -> Self {
        let mut this = *self;
        this.frobenius_map_in_place(power);
        this
    }

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();

        for i in ark_ff::BitIteratorBE::without_leading_zeros(exp) {
            res.square_in_place();

            if i {
                res *= self;
            }
        }
        res
    }

    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
        let mut res = Self::one();
        for (pow, bit) in ark_ff::BitIteratorLE::without_trailing_zeros(exp).enumerate() {
            if bit {
                res *= powers_of_2.get(pow)?;
            }
        }
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::BigInteger;
    use ark_ff::FftField;
    use ark_ff::PrimeField;
    use ark_ff::UniformRand;
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

    fn serialize_to_blst<const N: usize>(e: ark_ff::BigInt<N>) -> blstrs::Fp {
        let mut blst_buffer = [0u8; 48];
        //e.serialize_compressed(&mut blst_buffer[..]).unwrap();
        // blstrs::Fp::from_bytes_le(&blst_buffer).unwrap()
        blst_buffer.copy_from_slice(&e.to_bytes_le());
        blstrs::Fp::from_bytes_le(&blst_buffer).unwrap()
    }
    fn print_info<F: PrimeField>() {
        println!("MODULUS: {}", print_slice_hex(F::MODULUS.as_ref()));
        println!(
            "MODULUS_MINUS_ONE_DIV_TWO: {}",
            print_slice_hex(F::MODULUS_MINUS_ONE_DIV_TWO.as_ref())
        );
        println!(
            "TWO_ADIC_ROOT_OF_UNITY : {}",
            print_slice_hex(
                <F as FftField>::TWO_ADIC_ROOT_OF_UNITY
                    .into_bigint()
                    .as_ref()
            )
        );
    }

    #[test]
    fn print_info_fq() {
        print_info::<ark_bls12_381::Fq>();
    }

    #[test]
    fn test_fq_modulus() {
        print_info_fq();
        const slice_mod_minus_one_div_two: [u64; 6] = [
            0xa1fafffffffe5557,
            0x995bfff976a3fffe,
            0x3f41d24d174ceb4,
            0xf6547998c1995dbd,
            0x778a468f507a6034,
            0x20559931f7f8103,
        ];

        const modmodt: super::Fp = super::Fp(blstrs::Fp(blst_fp {
            l: slice_mod_minus_one_div_two,
        }));
        let manual_blst = modmodt.to_bytes_le();
        let serialized_from_arkworks =
            serialize_to_blst(ark_bls12_381::Fq::MODULUS_MINUS_ONE_DIV_TWO);
        let arkwork_version = ark_bls12_381::Fq::MODULUS_MINUS_ONE_DIV_TWO.to_bytes_le();
        assert_eq!(
            unsafe {
                std::slice::from_raw_parts(manual_blst.as_ptr() as *const _ as *const u8, 48)
            },
            arkwork_version.as_slice()
        );
        assert_eq!(manual_blst, serialized_from_arkworks.to_bytes_le());
        println!("MANUAL: 0x{}", hex::encode(manual_blst));
        println!(
            "SERIALIZED: 0x{}",
            hex::encode(serialized_from_arkworks.to_bytes_le())
        );
        println!("ARKWORKS: 0x{}", hex::encode(arkwork_version));

        println!(
            "slice from blst serialized: {}",
            print_slice_hex(serialized_from_arkworks.0.l.as_ref())
        );

        // ----- just  random stuff
        //let r1 = ark_bls12_381::Fq::rand(&mut rand::thread_rng());
        //let bytes_len = 48;
        //let r1_bytes = unsafe {
        //    std::slice::from_raw_parts(
        //        r1.into_bigint().0.as_ptr() as *const _ as *const u8,
        //        bytes_len,
        //    )
        //};
        //let mut blst_slice = [0u8; 48];
        //blst_slice.copy_from_slice(r1_bytes);
        //let r2 = blstrs::Fp::from_bytes_le(&blst_slice).unwrap();
        //let mut v1 = Vec::new();
        //r1.serialize_compressed(&mut v1).unwrap();
        //let arkworks_random = hex::encode(v1);
        //let blst_random = hex::encode(r2.to_bytes_le());
        //println!("arkworks random: {}", arkworks_random);
        //println!("blst random: {}", blst_random);
    }

    #[test]
    fn fp_tests() {
        let r = Fp::rand(&mut rand::thread_rng());
        let s = Fp::rand(&mut rand::thread_rng());
        let rps = r + s;
        assert!(rps.neg() + rps == Fp::zero());
        let spr = s + r;
        assert_eq!(rps, spr);

        let rps = r * s;
        assert!(rps.div(rps) == Fp::one());
        let spr = s * r;
        assert_eq!(rps, spr);

        let mut buff = Vec::new();
        r.serialize_compressed(&mut buff).unwrap();
        let r2 = Fp::deserialize_compressed(&buff[..]).unwrap();
        assert_eq!(r, r2);
    }
}
