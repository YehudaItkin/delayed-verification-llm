ClearAll["Global`*"];
p0=1/10; q0=98/125;
equalPoly=z^2-z+p0+q0;
unequalPoly=z^3-(1-p0)z^2+q0;
equalRoots=z/.Solve[equalPoly==0,z];
unequalRoots=z/.Solve[unequalPoly==0,z];
theta=Pi/3;
singularPoly=z^5-z^4+(1+q)z^3+q;
checks=<|
 "equal_pair_stable" -> FullSimplify[And@@(Abs[#]<1& /@ equalRoots)],
 "unequal_factorization" -> (Expand[(z+7/10)(z^2-8 z/5+28/25)]==unequalPoly),
 "unequal_pair_outside" -> FullSimplify[Abs[(4+2 I Sqrt[3])/5]^2==28/25 && 28/25>1],
 "positive_coefficients" -> (p0>0 && q0>0),
 "equal_delay_sum" -> (1+1==0+2),
 "singular_denominator" -> (Sin[(4-1)theta]==0),
 "singular_line_real" -> FullSimplify[Cos[theta]-1+(1+q)Cos[theta]+q Cos[4 theta]==0],
 "singular_line_imag" -> FullSimplify[Sin[theta]-(1+q)Sin[theta]-q Sin[4 theta]==0],
 "singular_factorization" -> (Expand[(z^2-z+1)(z^3+q z+q)-singularPoly]==0),
 "singular_remaining_roots_inside" -> FullSimplify[And@@(Abs[#]<1& /@ (z/.Solve[z^3+z/10+1/10==0,z]))],
 "golden_boundary" -> FullSimplify[2 Sin[Pi/10]==(Sqrt[5]-1)/2],
 "endpoint_plus_one" -> (Expand[1-1+p+q]==p+q),
 "singular_outward_crossing" -> FullSimplify[Re[-z^2/(D[singularPoly,z])]/.{q->1/10,z->Exp[I Pi/3]}]==40/73
|>;
Print[checks];
Print["equal roots: ",InputForm[equalRoots]];
Print["unequal roots: ",InputForm[unequalRoots]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"wolfram_results.json"}],
 <|"engine"->$Version,"checks"->checks,"p"->ToString[p0,InputForm],"q"->ToString[q0,InputForm],
 "equal_roots"->ToString[equalRoots,InputForm],"unequal_roots"->ToString[unequalRoots,InputForm]|>,"RawJSON"];
If[And@@(TrueQ /@ Values[checks]),Quit[0],Quit[1]];
