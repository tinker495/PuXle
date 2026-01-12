(define (domain test-constants)
  (:requirements :strips :typing)
  (:types item)
  (:constants
    c1 c2 - item
  )
  (:predicates
    (has ?x - item)
  )
  (:action pick
    :parameters (?x - item)
    :precondition ()
    :effect (has ?x)
  )
)
