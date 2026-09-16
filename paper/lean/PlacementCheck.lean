import Std

/- Exact certificates for the coherence/static-error counterexample.
   Uses rational arithmetic and Lean's kernel, without Mathlib.
   The abstract theorem only translates a supplied reduction guarantee;
   it does not assume or prove the full greedy/submodularity theorem. -/
namespace PlacementCheck
set_option maxRecDepth 100000
set_option maxHeartbeats 8000000

abbrev Mat := Fin 6 → Fin 6 → Rat
def ofRows (rows : Array (Array Rat)) : Mat :=
  fun i j => (rows.getD i.val #[]).getD j.val 0
def mul (a b : Mat) : Mat := fun i j =>
  ((List.finRange 6).map (fun k => a i k * b k j)).foldl (· + ·) 0
def trace (a : Mat) : Rat :=
  ((List.finRange 6).map (fun i => a i i)).foldl (· + ·) 0
def identity : Mat := fun i j => if i = j then 1 else 0

def lap : Mat := ofRows #[#[4, 0, -1, -1, -1, -1],
    #[0, 1, -1, 0, 0, 0],
    #[-1, -1, 3, 0, 0, -1],
    #[-1, 0, 0, 3, -1, -1],
    #[-1, 0, 0, -1, 2, 0],
    #[-1, 0, -1, -1, 0, 3]]

def matrixA : Mat := fun i j => lap i j +
  if i = j then 1/2 + (if i.val = 0 ∨ i.val = 1 then 2 else 0) else 0
def matrixB : Mat := fun i j => lap i j +
  if i = j then 1/2 + (if i.val = 1 ∨ i.val = 4 then 2 else 0) else 0

def inverseA : Mat := ofRows #[#[15994/65137, 2296/65137, 8036/65137, 10396/65137, 10556/65137, 9836/65137],
    #[2296/65137, 20774/65137, 7572/65137, 2144/65137, 1776/65137, 3432/65137],
    #[8036/65137, 7572/65137, 26502/65137, 7504/65137, 6216/65137, 12012/65137],
    #[10396/65137, 2144/65137, 7504/65137, 30134/65137, 16212/65137, 13724/65137],
    #[10556/65137, 1776/65137, 6216/65137, 16212/65137, 36762/65137, 9424/65137],
    #[9836/65137, 3432/65137, 12012/65137, 13724/65137, 9424/65137, 28774/65137]]
def inverseB : Mat := ofRows #[#[30626/77409, 4312/77409, 15092/77409, 16876/77409, 10556/77409, 17884/77409],
    #[4312/77409, 24902/77409, 9748/77409, 3104/77409, 1648/77409, 4904/77409],
    #[15092/77409, 9748/77409, 34118/77409, 10864/77409, 5768/77409, 17164/77409],
    #[16876/77409, 3104/77409, 10864/77409, 35414/77409, 11620/77409, 18044/77409],
    #[10556/77409, 1648/77409, 5768/77409, 11620/77409, 22130/77409, 7984/77409],
    #[17884/77409, 4904/77409, 17164/77409, 18044/77409, 7984/77409, 37286/77409]]

theorem inverse_certificate_A : ∀ i j, mul matrixA inverseA i j = identity i j ∧
    mul inverseA matrixA i j = identity i j := by decide +kernel

theorem inverse_certificate_B : ∀ i j, mul matrixB inverseB i j = identity i j ∧
    mul inverseB matrixB i j = identity i j := by decide +kernel

theorem exact_metrics_A : trace inverseA = 158940/65137 ∧
    trace (mul inverseA inverseA) = 6961553176/4242828769 := by decide +kernel

theorem exact_metrics_B : trace inverseB = 61492/25803 ∧
    trace (mul inverseB inverseB) = 1109382616/665794809 := by decide +kernel

theorem opposite_rankings : trace inverseB < trace inverseA ∧
    trace (mul inverseA inverseA) < trace (mul inverseB inverseB) := by decide +kernel

open Lean.Grind Std in
/-- Algebraic consequence of a reduction guarantee in any linearly ordered commutative ring.
    For q = 1 - 1/e this is Hg ≤ H0/e + (1 - 1/e) Ho. -/
theorem reduction_to_coherence_bound {K : Type} [CommRing K] [LE K] [LT K]
    [LawfulOrderLT K] [IsLinearOrder K] [OrderedRing K]
    (h0 hg ho q : K) (h : h0 - hg ≥ q * (h0 - ho)) :
    hg ≤ (1-q)*h0 + q*ho := by grind

#print axioms inverse_certificate_A
#print axioms inverse_certificate_B
#print axioms exact_metrics_A
#print axioms exact_metrics_B
#print axioms opposite_rankings
#print axioms reduction_to_coherence_bound
end PlacementCheck
