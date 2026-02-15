(define (domain test-disjunctive)
  (:requirements :strips :typing :disjunctive-preconditions)
  (:types loc)
  (:predicates
    (at ?x - loc)
    (safe ?x - loc)
  )
  (:action move
    :parameters (?x - loc)
    :precondition (or (at ?x) (safe ?x))
    :effect (at ?x)
  )
)
