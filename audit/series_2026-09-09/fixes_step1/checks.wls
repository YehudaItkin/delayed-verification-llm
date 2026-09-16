ClearAll["Global`*"];
seq = {0, 0};
Do[AppendTo[seq, Max[0, Last[seq] - Clip[seq[[-2]], {-2, 2}] + 1]], {18}];
checks = <|
  "positive_forcing_leaves_zero" -> FullSimplify[Max[0, g] == g && g > 0, Assumptions -> g > 0],
  "unforced_step_nonincreasing" -> FullSimplify[0 <= Max[0, p - alpha sig] <= p, Assumptions -> p >= 0 && alpha > 0 && sig >= 0],
  "exact_period_six" -> (seq == Take[Flatten[ConstantArray[{0, 0, 1, 2, 2, 1}, 4]], 20]),
  "grid_spacing" -> ((33/20 - 7/20)/39 == 1/30),
  "grid_lower" -> (7/20 + 19/30 == 59/60),
  "grid_upper" -> (7/20 + 20/30 == 61/60)
|>;
Print[checks];
Export[FileNameJoin[{DirectoryName[$InputFileName], "wolfram_results.json"}],
  <|"engine" -> $Version, "checks" -> checks, "periodic_sequence" -> seq|>, "RawJSON"];
If[And @@ (TrueQ /@ Values[checks]), Quit[0], Quit[1]];
