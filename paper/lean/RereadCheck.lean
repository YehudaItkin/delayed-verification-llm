import Std

/- Algebraic witnesses for the September 15 reread.
   These do not formalize stability or empirically identify an LLM update law. -/
namespace RereadCheck
open Lean.Grind
abbrev Mat := Fin 3 → Fin 3 → Rat
def ofRows (rows : Array (Array Rat)) : Mat :=
  fun i j => (rows.getD i.val #[]).getD j.val 0
def mul (a b : Mat) : Mat := fun i j =>
  ((List.finRange 3).map (fun k => a i k * b k j)).foldl (· + ·) 0
def transpose (a : Mat) : Mat := fun i j => a j i
def directed : Mat := ofRows #[#[2, -1, 0], #[0, 2, -1], #[-1, 0, 2]]
def gram : Mat := ofRows #[#[5, -2, -2], #[-2, 5, -2], #[-2, -2, 5]]
theorem directed_nonsymmetric : directed 0 1 ≠ directed 1 0 := by decide +kernel
theorem directed_normal : ∀ i j,
    mul directed (transpose directed) i j = gram i j ∧
    mul (transpose directed) directed i j = gram i j := by decide +kernel
theorem fixed_peer_expansion (e eta w f k h : Rat) :
    e + eta*w*(f-e) - eta*k*h = (1-eta*w)*e - eta*k*h + eta*w*f := by grind
theorem omitted_diagonal (e eta w f k h : Rat) :
    (e + eta*w*(f-e) - eta*k*h) - (e + eta*w*f - eta*k*h) = -eta*w*e := by grind
theorem concrete_peer_mismatch :
    (1 + (1/2 : Rat)*(2-1)) = 3/2 ∧ (1 + (1/2 : Rat)*2) = 2 := by decide +kernel
#print axioms directed_nonsymmetric
#print axioms directed_normal
#print axioms fixed_peer_expansion
#print axioms omitted_diagonal
#print axioms concrete_peer_mismatch
end RereadCheck
