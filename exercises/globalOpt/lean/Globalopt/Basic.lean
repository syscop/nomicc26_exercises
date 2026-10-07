import Mathlib

set_option linter.style.header false

/-!
# TutorialGO: machine-checked facts

Lean/Mathlib proofs of the algebraic facts behind the answers in `../answers.tex`.
Everything is stated over `ℝ` with rational data. The certificates (sums of squares,
kernel vectors, bound-factor products) were found with sympy and are only *checked* here.
-/

namespace TutorialGO

/-! ## Question 1: one lifted variable `X₁₁` suffices for `Q₁` and `Q₂` -/

/-- `xᵀ Q₁ x` -/
def q1 (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  -2 * x₁ ^ 2 - 2 * x₁ * x₂ + 2 * x₂ ^ 2 + 3 * x₃ ^ 2 - 2 * x₃ * x₄ + 2 * x₄ ^ 2

/-- `xᵀ Q₂ x` -/
def q2 (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  -2 * x₁ ^ 2 + 2 * x₁ * x₂ + 2 * x₂ ^ 2 - 2 * x₂ * x₃ + 3 * x₃ ^ 2 - 2 * x₃ * x₄ + 2 * x₄ ^ 2

/-- Convex split of `Q₁`: `Q₁ + 5/2 e₁e₁ᵀ` is a sum of squares, so the only nonconvex
term is `-5/2 x₁²`. -/
theorem q1_split (x₁ x₂ x₃ x₄ : ℝ) :
    q1 x₁ x₂ x₃ x₄ =
      (2 * (x₂ - x₁ / 2) ^ 2 + 2 * (x₄ - x₃ / 2) ^ 2 + 5 / 2 * x₃ ^ 2) - 5 / 2 * x₁ ^ 2 := by
  unfold q1; ring

/-- Convex split of `Q₂` with the smallest shift `d* = 21/8`. -/
theorem q2_split (x₁ x₂ x₃ x₄ : ℝ) :
    q2 x₁ x₂ x₃ x₄ =
      (2 * (x₄ - x₃ / 2) ^ 2 + 5 / 2 * (x₃ - 2 / 5 * x₂) ^ 2 + 8 / 5 * (x₂ + 5 / 8 * x₁) ^ 2)
        - 21 / 8 * x₁ ^ 2 := by
  unfold q2; ring

/-- `d* = 21/8` is the smallest shift: for any smaller `d` the kernel vector `(-8, 5, 2, 1)`
of `Q₂ + d* e₁e₁ᵀ` makes `xᵀ (Q₂ + d e₁e₁ᵀ) x` negative. -/
theorem q2_shift_smallest (d : ℝ) (hd : d < 21 / 8) :
    q2 (-8) 5 2 1 + d * (-8) ^ 2 < 0 := by
  norm_num [q2]; linarith

/-- The secant of `x₁²` on `[-1, 1]` is `1`. -/
theorem sq_le_one_of_box {x : ℝ} (h₁ : -1 ≤ x) (h₂ : x ≤ 1) : x ^ 2 ≤ 1 := by
  nlinarith

/-- The relaxation "convex part `- 5/2 X₁₁`, `X₁₁ ≤ 1`" proves the global minimum of `xᵀQ₁x`
on `[-1,1]⁴` (with `b = 0`) ... -/
theorem q1_lower (x₁ x₂ x₃ x₄ : ℝ) (h₁ : -1 ≤ x₁) (h₂ : x₁ ≤ 1) :
    -5 / 2 ≤ q1 x₁ x₂ x₃ x₄ := by
  rw [q1_split]
  have := sq_le_one_of_box h₁ h₂
  nlinarith [sq_nonneg (x₂ - x₁ / 2), sq_nonneg (x₄ - x₃ / 2), sq_nonneg x₃]

/-- ... and it is attained. -/
theorem q1_attained : q1 (-1) (-1 / 2) 0 0 = -5 / 2 := by norm_num [q1]

theorem q2_lower (x₁ x₂ x₃ x₄ : ℝ) (h₁ : -1 ≤ x₁) (h₂ : x₁ ≤ 1) :
    -21 / 8 ≤ q2 x₁ x₂ x₃ x₄ := by
  rw [q2_split]
  have := sq_le_one_of_box h₁ h₂
  nlinarith [sq_nonneg (x₄ - x₃ / 2), sq_nonneg (x₃ - 2 / 5 * x₂), sq_nonneg (x₂ + 5 / 8 * x₁)]

theorem q2_attained : q2 (-1) (5 / 8) (1 / 4) (1 / 8) = -21 / 8 := by norm_num [q2]

/-! ## Question 2: the single bilinear term `w = u v` -/

/-- Interval bounds of `u = 1 - x₁ + 2x₂ - 2x₄` and `v = 4x₂ - 5x₃ + x₄ + 2` on `[-1,1]⁴`. -/
theorem uv_bounds (x₁ x₂ x₃ x₄ : ℝ)
    (h₁ : -1 ≤ x₁ ∧ x₁ ≤ 1) (h₂ : -1 ≤ x₂ ∧ x₂ ≤ 1)
    (h₃ : -1 ≤ x₃ ∧ x₃ ≤ 1) (h₄ : -1 ≤ x₄ ∧ x₄ ≤ 1) :
    (-4 ≤ 1 - x₁ + 2 * x₂ - 2 * x₄ ∧ 1 - x₁ + 2 * x₂ - 2 * x₄ ≤ 6) ∧
    (-8 ≤ 4 * x₂ - 5 * x₃ + x₄ + 2 ∧ 4 * x₂ - 5 * x₃ + x₄ + 2 ≤ 12) := by
  obtain ⟨_, _⟩ := h₁; obtain ⟨_, _⟩ := h₂; obtain ⟨_, _⟩ := h₃; obtain ⟨_, _⟩ := h₄
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> linarith

/-- The four McCormick inequalities for `w = u v` on `[-4, 6] × [-8, 12]`. -/
theorem mccormick (u v : ℝ) (hu₁ : -4 ≤ u) (hu₂ : u ≤ 6) (hv₁ : -8 ≤ v) (hv₂ : v ≤ 12) :
    -8 * u - 4 * v - 32 ≤ u * v ∧ 12 * u + 6 * v - 72 ≤ u * v ∧
    u * v ≤ 12 * u - 4 * v + 48 ∧ u * v ≤ -8 * u + 6 * v + 48 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ u + 4) (by linarith : (0 : ℝ) ≤ v + 8)]
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 6 - u) (by linarith : (0 : ℝ) ≤ 12 - v)]
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ u + 4) (by linarith : (0 : ℝ) ≤ 12 - v)]
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 6 - u) (by linarith : (0 : ℝ) ≤ v + 8)]

/-! ## Question 3: the quartic `f = -x₁²x₂² + 2x₁x₂³ - x₂⁴` -/

def f (x₁ x₂ : ℝ) : ℝ := -x₁ ^ 2 * x₂ ^ 2 + 2 * x₁ * x₂ ^ 3 - x₂ ^ 4

theorem f_eq (x₁ x₂ : ℝ) : f x₁ x₂ = -(x₁ * x₂ - x₂ ^ 2) ^ 2 := by unfold f; ring

/-- The sheet's linearisation of `(1 - x₁)²(1 - x₂)² ≥ 0` (before replacing monomials by `X`). -/
theorem sheet_bound_factor (x₁ x₂ : ℝ) :
    (1 - x₁) ^ 2 * (1 - x₂) ^ 2 =
      1 - 2 * x₁ - 2 * x₂ + 4 * (x₁ * x₂) + x₁ ^ 2 + x₂ ^ 2
        - 2 * (x₁ * x₂ ^ 2) - 2 * (x₂ * x₁ ^ 2) + x₁ ^ 2 * x₂ ^ 2 := by ring

/-- Exact range of `w = x₁x₂ - x₂²` on `[-1,1]²` is `[-2, 1/4]`. -/
theorem w_range (x₁ x₂ : ℝ) (h₁ : -1 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : -1 ≤ x₂) (h₄ : x₂ ≤ 1) :
    -2 ≤ x₁ * x₂ - x₂ ^ 2 ∧ x₁ * x₂ - x₂ ^ 2 ≤ 1 / 4 := by
  constructor
  · nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 1 + x₁) (by linarith : (0 : ℝ) ≤ 1 + x₂),
               mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - x₁) (by linarith : (0 : ℝ) ≤ 1 - x₂),
               mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - x₂) (by linarith : (0 : ℝ) ≤ 1 + x₂)]
  · rcases le_total 0 x₂ with h | h
    · nlinarith [sq_nonneg (x₂ - 1 / 2), mul_nonneg h (by linarith : (0 : ℝ) ≤ 1 - x₁)]
    · nlinarith [sq_nonneg (x₂ + 1 / 2),
                 mul_nonneg (by linarith : (0 : ℝ) ≤ -x₂) (by linarith : (0 : ℝ) ≤ 1 + x₁)]

theorem w_range_attained : (-1 : ℝ) * 1 - 1 ^ 2 = -2 ∧ (1 : ℝ) * (1 / 2) - (1 / 2) ^ 2 = 1 / 4 := by
  norm_num

/-- The secant of the concave `-w²` on `[-2, 1/4]` is a valid underestimator. -/
theorem secant (w : ℝ) (h₁ : -2 ≤ w) (h₂ : w ≤ 1 / 4) : 7 / 4 * w - 1 / 2 ≤ -w ^ 2 := by
  nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ w + 2) (by linarith : (0 : ℝ) ≤ 1 / 4 - w)]

/-- Global minimum of `f` on `[-1,1]²` is `-4`, attained at `(-1, 1)`. -/
theorem f_lower (x₁ x₂ : ℝ) (h₁ : -1 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : -1 ≤ x₂) (h₄ : x₂ ≤ 1) :
    -4 ≤ f x₁ x₂ := by
  obtain ⟨hl, hu⟩ := w_range x₁ x₂ h₁ h₂ h₃ h₄
  rw [f_eq]
  nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ x₁ * x₂ - x₂ ^ 2 + 2)
                        (by linarith : (0 : ℝ) ≤ 2 - (x₁ * x₂ - x₂ ^ 2))]

theorem f_attained : f (-1) 1 = -4 := by norm_num [f]

/-! ## Question 4: `min / max xᵀQx` s.t. `xᵀx = 1` -/

/-- At a KKT point `Qx = μx`, `xᵀx = 1`, the objective equals the multiplier. -/
theorem kkt_value {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) (μ : ℝ)
    (h : Q.mulVec x = μ • x) (hx : x ⬝ᵥ x = 1) : x ⬝ᵥ Q.mulVec x = μ := by
  rw [h, dotProduct_smul, hx, smul_eq_mul, mul_one]

def Q4 : Matrix (Fin 4) (Fin 4) ℝ := !![4, -1, 0, 0; -1, 4, -2, 0; 0, -2, 3, 2; 0, 0, 2, 1]

theorem Q4_charpoly (μ : ℝ) :
    (Q4 - μ • (1 : Matrix (Fin 4) (Fin 4) ℝ)).det =
      μ ^ 4 - 12 * μ ^ 3 + 42 * μ ^ 2 - 32 * μ - 31 := by
  have : Q4 - μ • (1 : Matrix (Fin 4) (Fin 4) ℝ) =
      !![4 - μ, -1, 0, 0; -1, 4 - μ, -2, 0; 0, -2, 3 - μ, 2; 0, 0, 2, 1 - μ] := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [Q4]
  rw [this, Matrix.det_succ_row_zero]
  simp [Fin.sum_univ_succ, Matrix.det_succ_row_zero, Fin.succAbove]
  ring

/-- `xᵀ Q x` written out. -/
def qQ (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  4 * x₁ ^ 2 - 2 * x₁ * x₂ + 4 * x₂ ^ 2 - 4 * x₂ * x₃ + 3 * x₃ ^ 2 + 4 * x₃ * x₄ + x₄ ^ 2

/-- Lower bracket: `Q + 0.535 I ⪰ 0` (rational LDLᵀ certificate), so `λ_min ≥ -0.535`. -/
theorem lam_min_ge (x₁ x₂ x₃ x₄ : ℝ) :
    -535 / 1000 * (x₁ ^ 2 + x₂ ^ 2 + x₃ ^ 2 + x₄ ^ 2) ≤ qQ x₁ x₂ x₃ x₄ := by
  unfold qQ
  nlinarith [mul_nonneg (by norm_num : (0 : ℝ) ≤ 907 / 200) (sq_nonneg (x₁ - 200 / 907 * x₂)),
             mul_nonneg (by norm_num : (0 : ℝ) ≤ 782649 / 181400)
               (sq_nonneg (x₂ - 362800 / 782649 * x₃)),
             mul_nonneg (by norm_num : (0 : ℝ) ≤ 408212843 / 156529800)
               (sq_nonneg (x₃ + 313059600 / 408212843 * x₄)),
             mul_nonneg (by norm_num : (0 : ℝ) ≤ 97502801 / 81642568600) (sq_nonneg x₄)]

/-- Upper bracket: the Rayleigh quotient of `(-60, -271, -585, 762)` is below `-0.534`. -/
theorem lam_min_le :
    qQ (-60) (-271) (-585) 762 <
      -534 / 1000 * ((-60) ^ 2 + (-271) ^ 2 + (-585) ^ 2 + 762 ^ 2) := by
  norm_num [qQ]

/-- Upper bracket for the maximum: `6.151 I - Q ⪰ 0`. -/
theorem lam_max_le (x₁ x₂ x₃ x₄ : ℝ) :
    qQ x₁ x₂ x₃ x₄ ≤ 6151 / 1000 * (x₁ ^ 2 + x₂ ^ 2 + x₃ ^ 2 + x₄ ^ 2) := by
  unfold qQ
  nlinarith [mul_nonneg (by norm_num : (0 : ℝ) ≤ 2151 / 1000) (sq_nonneg (x₁ + 1000 / 2151 * x₂)),
             mul_nonneg (by norm_num : (0 : ℝ) ≤ 3626801 / 2151000)
               (sq_nonneg (x₂ + 4302000 / 3626801 * x₃)),
             mul_nonneg (by norm_num : (0 : ℝ) ≤ 2824049951 / 3626801000)
               (sq_nonneg (x₃ - 7253602000 / 2824049951 * x₄)),
             mul_nonneg (by norm_num : (0 : ℝ) ≤ 39477297601 / 2824049951000) (sq_nonneg x₄)]

/-- Lower bracket for the maximum: the Rayleigh quotient of `(-326, 701, -591, -229)`
exceeds `6.150`. -/
theorem lam_max_ge :
    6150 / 1000 * ((-326) ^ 2 + 701 ^ 2 + (-591) ^ 2 + (-229) ^ 2) <
      qQ (-326) 701 (-591) (-229) := by
  norm_num [qQ]

end TutorialGO
