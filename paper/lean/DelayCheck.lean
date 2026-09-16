import Std

/- Algebraic certificates for the two-delay audit. These prove identities and
   rational inequalities, not the entire complex-root stability theorem. -/
namespace DelayCheck
open Lean.Grind

def p : Rat := 1/10
def q : Rat := 98/125

theorem same_delay_sum : 1 + 1 = (0 : Nat) + 2 := by decide

theorem equal_lag_gain : p > 0 ∧ q > 0 ∧ p + q = 221/250 ∧ p + q < 1 :=
  by decide +kernel

/-- Complex roots (1/2) ± i sqrt(1585)/50 have squared modulus below one. -/
theorem equal_root_certificates :
    (1/2 : Rat)^2 + 1585*(1/50)^2 = p+q ∧
    (1/2 : Rat)^2 + 1585*(1/50)^2 < 1 ∧
    (1/2 : Rat)^2 - 1585*(1/50)^2 - 1/2 + (p+q) = 0 ∧
    2*(1/2 : Rat)*(1/50) - 1/50 = 0 := by decide +kernel

theorem unequal_factorization {K : Type} [Field K] [IsCharP K 0] (z : K) :
    z^3 - (9/10)*z^2 + 98/125 =
      (z+7/10)*(z^2-(8/5)*z+28/25) := by grind

/-- The real root -7/10 is inside, but (4 ± 2 i sqrt(3))/5 are outside. -/
theorem unequal_root_certificates :
    (-7/10 : Rat)^2 < 1 ∧
    (4/5 : Rat)^2 + 3*(2/5)^2 = 28/25 ∧
    (28/25 : Rat) > 1 ∧
    (4/5 : Rat)^2 - 3*(2/5)^2 - (8/5)*(4/5) + 28/25 = 0 ∧
    2*(4/5 : Rat)*(2/5) - (8/5)*(2/5) = 0 := by decide +kernel

theorem singular_line_factorization {K : Type} [CommRing K] (z q : K) :
    z^5-z^4+(1+q)*z^3+q = (z^2-z+1)*(z^3+q*z+q) := by grind

/-- Roots (1 ± i sqrt(3))/2 of the singular-line factor have unit modulus. -/
theorem singular_root_certificates :
    (1/2 : Rat)^2 + 3*(1/2)^2 = 1 ∧
    (1/2 : Rat)^2 - 3*(1/2)^2 - 1/2 + 1 = 0 ∧
    2*(1/2 : Rat)*(1/2) - 1/2 = 0 ∧
    (1/10 : Rat) + 1/10 < 1 := by decide +kernel

#print axioms same_delay_sum
#print axioms equal_lag_gain
#print axioms equal_root_certificates
#print axioms unequal_factorization
#print axioms unequal_root_certificates
#print axioms singular_line_factorization
#print axioms singular_root_certificates
end DelayCheck
