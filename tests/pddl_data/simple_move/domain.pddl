(define (domain simple-move)
  (:requirements :strips :typing)
  (:types location)
  (:predicates
    (at ?x - location)
    (connected ?from - location ?to - location)
  )
  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (at ?from) (connected ?from ?to))
    :effect (and (not (at ?from)) (at ?to))
  )
)
