import Std
namespace ControllerCheck
open Lean.Grind
theorem legacy_lag (t r : Nat) (hr : r ≥ 1) : t-r = (t-1)-(r-1) := by omega
theorem theory_index (t delay : Nat) : (t+1)-1-delay = t-delay := by omega
theorem imposed_error_update {K : Type} [CommRing K] (current stale truth gain : K) :
    (current-gain*(stale-truth))-truth = (current-truth)-gain*(stale-truth) := by grind
theorem high_gain_zero_lag_damped : (1-(3/2 : Rat))^2 = 1/4 ∧ (1/4 : Rat) < 1 := by decide +kernel
#print axioms legacy_lag
#print axioms theory_index
#print axioms imposed_error_update
#print axioms high_gain_zero_lag_damped
end ControllerCheck
