(define (domain test-zero-arity)
  (:requirements :strips)
  (:predicates
    (light-on)
  )
  (:action turn-on
    :parameters ()
    :precondition (not (light-on))
    :effect (light-on)
  )
)
