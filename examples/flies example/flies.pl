foil_numeric_predicates([]).
foil_predicates([flies/1,bird/1, penguin/1, cat/1, superpenguin/1, plane/1, damaged/1]).                                
foil_cwa(true).                                            
foil_use_negations(false).                                 
foil_det_lit_bound(0).

bird(X) :- penguin(X).
penguin(X) :- superpenguin(X).
bird(a).
bird(b).
penguin(c).
penguin(d).
superpenguin(e).
superpenguin(f).
cat(c1).
plane(g).
plane(h).
plane(k).
plane(m).
damaged(k).
damaged(m).

flies(a).
flies(b).

flies(e).
flies(f).

flies(g).
flies(h).



