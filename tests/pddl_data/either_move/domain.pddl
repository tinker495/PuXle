(define (domain either-move)
  (:requirements :strips :typing)
  (:types room hall)
  (:predicates
    (at ?x - (either room hall))
  )
  (:action mark
    :parameters (?x - (either room hall))
    :precondition (at ?x)
    :effect (at ?x)
  )
)
