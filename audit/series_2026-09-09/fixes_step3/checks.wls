ClearAll["Global`*"];
counter = z^3 + z^2/2 + 1/2;
radialNumerator = (d + 1) + d a^2 - (2 d + 1) a c;
unit = c + I s;
checks = <|
  "negative_a_factorization" -> (Expand[(z + 1) (2 z^2 - z + 1)/2] == counter),
  "negative_a_other_roots_inside" -> FullSimplify[Abs[(1 + I Sqrt[7])/4] == 1/Sqrt[2]],
  "delta_one_boundary_constant" -> FullSimplify[Abs[(a + I Sqrt[4 - a^2])/2] == 1, Assumptions -> 0 < a < 1],
  "zero_a_all_roots_on_circle" -> And @@ Flatten[Table[FullSimplify[Abs[Exp[I (2 j + 1) Pi/(d + 1)]] == 1], {d, 1, 12}, {j, 0, d}]],
  "phase_derivative" -> FullSimplify[ComplexExpand[Im[I unit/(unit - a)]] == (1 - a c)/(1 - 2 a c + a^2), Assumptions -> 0 < a < 1 && c^2 + s^2 == 1 && s > 0],
  "phase_derivative_positive" -> FullSimplify[(1 - a c)/(1 - 2 a c + a^2) > 0, Assumptions -> 0 < a < 1 && -1 <= c <= 1],
  "radial_crossing_identity" -> FullSimplify[ComplexExpand[Re[(unit - a)/((d + 1) unit - d a)]] == radialNumerator/(((d + 1) c - d a)^2 + (d + 1)^2 (1 - c^2)), Assumptions -> 0 < a < 1 && c^2 + s^2 == 1 && s > 0 && d >= 1],
  "radial_crossing_positive" -> FullSimplify[radialNumerator > 0, Assumptions -> 0 < a < 1 && -1 <= c <= 1 && d >= 1],
  "radial_lower_bound_factorization" -> (Expand[radialNumerator - (1 - a) (d + 1 - d a) - (2 d + 1) a (1 - c)] == 0),
  "delta_two_formula" -> FullSimplify[((Sqrt[a^2 + 4] - a)/2)^2 + a (Sqrt[a^2 + 4] - a)/2 == 1],
  "limit_a_one_sine_identity" -> FullSimplify[Sin[x]/Sin[(Pi - x)/2] == 2 Sin[x/2], Assumptions -> 0 < x < Pi],
  "zero_a_modulus_identity" -> FullSimplify[(b^(1/(d + 1)))^(d + 1) == b, Assumptions -> b > 0 && d >= 1]
|>;
Print[checks];
Export[FileNameJoin[{DirectoryName[$InputFileName], "wolfram_results.json"}],
  <|"engine" -> $Version, "checks" -> checks|>, "RawJSON"];
If[And @@ (TrueQ /@ Values[checks]), Quit[0], Quit[1]];
