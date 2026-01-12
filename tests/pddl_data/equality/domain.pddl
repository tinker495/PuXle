(define (domain test-equality)
  (:requirements :strips :typing :equality)
  (:types loc)
  (:predicates
    (at ?l - loc)
  )
  (:action move
    :parameters (?from ?to - loc)
    :precondition (and (at ?from) (not (= ?from ?to)))
    :effect (and (not (at ?from)) (at ?to))
  )
)
