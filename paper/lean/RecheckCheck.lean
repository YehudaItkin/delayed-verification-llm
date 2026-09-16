import Std
namespace RecheckCheck
open Lean.Grind
theorem gossip_factorization (z : Rat) :
    (z+2/3)*(z^2-5*z/3+10/9) = z^3-z^2+20/27 := by grind
theorem gossip_modulus_witness :
    (5/6 : Rat)^2 + 15/36 = 10/9 ∧ (10/9 : Rat) > 1 := by decide +kernel
theorem instantaneous_stable : (0 : Rat) < 1-20/27 ∧ (1-20/27 : Rat) < 1 := by decide +kernel
theorem noncommuting_entry : (2*0 + (-1)*2 : Rat) ≠ 1*(-1)+0*2 := by decide +kernel
theorem degroot_disagreement : (1/10 - 9/10 : Rat) = -4/5 := by decide +kernel
theorem lag_zero_only_sum (a b c x : Rat) : a*x+b*x+c = (a+b)*x+c := by grind
theorem f1_alias (t : Nat) (ht : t ≥ 1) : t-1 = t-1-0 := by omega
#print axioms gossip_factorization
#print axioms gossip_modulus_witness
#print axioms instantaneous_stable
#print axioms noncommuting_entry
#print axioms degroot_disagreement
#print axioms lag_zero_only_sum
#print axioms f1_alias
end RecheckCheck
