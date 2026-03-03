// Copyright (c) 2025, Ned Ganchovski
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use super::super::FloatType;
use super::Convergency;
use super::SearchError;

/// Illinois modification to the classical method
#[derive(Debug, PartialEq)]
enum Edge {
    /// Value is close to X1, reduce the Y1 weight
    EdgeX1,
    /// Value is in the middle of the interval
    NoEdge,
    /// Value is close to X2, reduce the Y2 weight
    EdgeX2,
}

/// Find a root of the function f(x) = 0 using a modification ofthe Modified Anderson-Bork's method:
// "Modified Anderson-Bjork’s method for solving non-linear equations in structural mechanics" 
// (https://doi.org/10.1088/1757-899X/1276/1/012010) by N Ganchovski and A Traykov. 
///
/// Pro
///
/// + Robust
/// + Fast
/// + No need for derivative function
///
/// Contra
///
/// - Needs initial bracketing
/// - A little complicated to implement
///
/// # Failures
/// ## NoBracketing
/// Initial values do not bracket the root.
/// ## NoConvergency
/// Algorithm cannot find a root within the given number of iterations.
/// # Examples
///
/// ```
/// use roots::SimpleConvergency;
/// use roots::find_root_modab;
///
/// let f = |x| { 1f64*x*x - 1f64 };
/// let mut convergency = SimpleConvergency { eps:1e-15f64, max_iter:30 };
///
/// let root1 = find_root_modab(10f64, 0f64, &f, &mut convergency);
/// // Returns approximately Ok(1);
///
/// let root2 = find_root_modab(-10f64, 0f64, &f, &mut 1e-15f64);
/// // Returns approximately Ok(-1);
/// ```
pub fn find_root_modab<F, Func>(a: F, b: F, mut f: Func, convergency: &mut dyn Convergency<F>) -> Result<F, SearchError>
where
    F: FloatType,
    Func: FnMut(F) -> F,
{
    let (mut x1, mut x2) = if a > b { (b, a) } else { (a, b) };
    let mut y1 = f(x1);
    if convergency.is_root_found(y1) {
        return Ok(x1);
    }
    let mut y2 = f(x2);
    if convergency.is_root_found(y2) {
        return Ok(x2);
    }
    if y1 * y2 > F::zero() {
        return Err(SearchError::NoBracketing);
    }

    let mut edge = Edge::NoEdge;
    let mut iter = 0;
    let mut bisection = true;
    let mut threshold = x2 - x1;

    let c = F::from(16i16);
    let two = F::from(2i16);

    loop {
        let x3: F;
        let y3: F;

        if bisection {
            x3 = (x1 + x2) / two;
            y3 = f(x3);
            let ym = (y1 + y2) / two;
            let r = F::one() - (ym / (y2 - y1)).abs();
            let k = r * r;

            if (ym - y3).abs() < k * (y3.abs() + ym.abs()) {
                bisection = false;
                threshold = (x2 - x1) * c;
            }
        } else {
            x3 = ((x1 * y2 - y1 * x2) / (y2 - y1)).clamp(x1, x2);
            y3 = f(x3);
            threshold = threshold / two;
        }

        if convergency.is_root_found(y3) {
            return Ok(x3);
        }
        if convergency.is_converged(x1, x2) {
            return Ok(x3);
        }

        if y1.signum() == y3.signum() {
            if edge == Edge::EdgeX1 {
                let m = F::one() - y3 / y1;
                if m <= F::zero() {
                    y2 = y2 / two;
                } else {
                    y2 = y2 * m;
                }
            } else if !bisection {
                edge = Edge::EdgeX1;
            }
            x1 = x3;
            y1 = y3;
        } else {
            if edge == Edge::EdgeX2 {
                let m = F::one() - y3 / y2;
                if m <= F::zero() {
                    y1 = y1 / two;
                } else {
                    y1 = y1 * m;
                }
            } else if !bisection {
                edge = Edge::EdgeX2;
            }
            x2 = x3;
            y2 = y3;
        }

        if x2 - x1 > threshold {
            bisection = true;
            edge = Edge::NoEdge;
        }

        iter += 1;
        if convergency.is_iteration_limit_reached(iter) {
            return Err(SearchError::NoConvergency);
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::*;
    use super::*;

    #[test]
    fn test_find_root_modab() {
        let f = |x| 1f64 * x * x - 1f64;
        let mut conv = debug_convergency::DebugConvergency::new(1e-15f64, 30);

        conv.reset();
        assert_float_eq!(
            1e-15f64,
            find_root_modab(10f64, 0f64, &f, &mut conv).ok().unwrap(),
            1f64
        );
        assert_eq!(8, conv.get_iter_count());

        conv.reset();
        assert_float_eq!(
            1e-15f64,
            find_root_modab(-10f64, 0f64, &f, &mut conv).ok().unwrap(),
            -1f64
        );
        assert_eq!(8, conv.get_iter_count());

        conv.reset();
        assert_eq!(
            find_root_modab(10f64, 20f64, &f, &mut conv),
            Err(SearchError::NoBracketing)
        );
        let result = find_root_modab(10f64, 20f64, &f, &mut conv);
        assert_eq!(result.unwrap_err().to_string(), "Bracketing Error");
        assert_eq!(0, conv.get_iter_count());
    }
}
