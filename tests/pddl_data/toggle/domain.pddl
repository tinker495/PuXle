(define (domain toggle)
  (:requirements :strips)
  (:predicates (on) (off))
  (:action flip-on
    :parameters ()
    :precondition (off)
    :effect (and (on) (not (off))))
  (:action flip-off
    :parameters ()
    :precondition (on)
    :effect (and (off) (not (on))))
)
