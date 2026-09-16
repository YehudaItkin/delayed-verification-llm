ClearAll["Global`*"];
edges = {{0,2},{0,3},{0,4},{0,5},{1,2},{2,5},{3,4},{3,5}};
adj = ConstantArray[0,{6,6}];
Do[adj[[edge[[1]]+1,edge[[2]]+1]]=1; adj[[edge[[2]]+1,edge[[1]]+1]]=1, {edge,edges}];
lap = DiagonalMatrix[Total[adj]]-adj;
ma = lap + IdentityMatrix[6]/2 + DiagonalMatrix[{2,2,0,0,0,0}];
mb = lap + IdentityMatrix[6]/2 + DiagonalMatrix[{0,2,0,0,2,0}];
qa=Inverse[ma]; qb=Inverse[mb];
metrics={{Tr[qa],Tr[qa.qa]},{Tr[qb],Tr[qb.qb]}};
q2=Inverse[{{2+s,-1},{-1,2+t}}];
checks=<|
 "matrix_a_positive_definite" -> And@@Table[Det[ma[[1;;j,1;;j]]]>0,{j,6}],
 "matrix_b_positive_definite" -> And@@Table[Det[mb[[1;;j,1;;j]]]>0,{j,6}],
 "inverse_certificates" -> (ma.qa==IdentityMatrix[6] && mb.qb==IdentityMatrix[6]),
 "coherence_prefers_b" -> (Tr[qb]<Tr[qa]),
 "static_mse_prefers_a" -> (Tr[qa.qa]<Tr[qb.qb]),
 "ou_covariance_equation" -> (ma.(qa/2)+(qa/2).ma==IdentityMatrix[6]),
 "mixed_derivative_identity" -> (FullSimplify[D[Tr[q2],s,t]-2 q2[[1,2]] (q2.q2)[[1,2]]]==0),
 "greedy_bound_rearrangement" -> (Expand[(h0-hg-q(h0-ho))-((1-q)h0+q ho-hg)]==0)
|>;
Print[checks]; Print["metrics: ",InputForm[metrics]];
Export[FileNameJoin[{DirectoryName[$InputFileName],"wolfram_results.json"}],
 <|"engine"->$Version,"checks"->checks,"edges"->edges,"laplacian"->lap,
 "inverse_a"->Map[ToString[#,InputForm]&,qa,{2}],
 "inverse_b"->Map[ToString[#,InputForm]&,qb,{2}],
 "metrics"->Map[ToString[#,InputForm]&,metrics,{2}]|>,"RawJSON"];
If[And@@(TrueQ /@ Values[checks]),Quit[0],Quit[1]];
