ClearAll["Global`*"];
path = {{1, -1, 0}, {-1, 2, -1}, {0, -1, 1}};
groundedOnce = path[[{2, 3}, {2, 3}]];
groundedTwice = path[[{2}, {2}]];
spectra = Sort[Eigenvalues[#]] & /@ {path, groundedOnce, groundedTwice};
soft = groundedOnce + DiagonalMatrix[{2, 0}];
softSpectrum = Sort[Eigenvalues[soft]];
zeroRoots = z /. Solve[z^2 - z + 1/2 == 0, z];
otherRoots = z /. Solve[z^2 + z + 1/2 == 0, z];
checks = <|
  "path_spectra_exact" -> (spectra == {{0, 1, 3}, {(3 - Sqrt[5])/2, (3 + Sqrt[5])/2}, {2}}),
  "hard_grounding_min_nondecreasing" -> FullSimplify[LessEqual @@ (First /@ spectra)],
  "hard_grounding_max_nonincreasing" -> FullSimplify[GreaterEqual @@ (Last /@ spectra)],
  "soft_pin_min_increases" -> FullSimplify[First[softSpectrum] > First[spectra[[2]]]],
  "soft_pin_max_increases" -> FullSimplify[Last[softSpectrum] > Last[spectra[[2]]]],
  "zero_mode_roots_strictly_stable" -> FullSimplify[And @@ (Abs[#] == 1/Sqrt[2] & /@ zeroRoots)],
  "ungrounded_K2_other_mode_stable" -> FullSimplify[And @@ (Abs[#] == 1/Sqrt[2] & /@ otherRoots)],
  "bias_equilibrium_identity" -> FullSimplify[e == (1 - eta mu) e - eta kappa e + eta g /. e -> g/(mu + kappa), Assumptions -> mu >= 0 && kappa > 0]
|>;
Print[checks];
Export[FileNameJoin[{DirectoryName[$InputFileName], "wolfram_results.json"}],
  <|"engine" -> $Version, "checks" -> checks,
    "hard_grounding_spectra" -> (ToString[#, InputForm] & /@ spectra),
    "soft_pin_spectrum" -> ToString[softSpectrum, InputForm],
    "zero_mode_roots" -> ToString[zeroRoots, InputForm]|>, "RawJSON"];
If[And @@ (TrueQ /@ Values[checks]), Quit[0], Quit[1]];
