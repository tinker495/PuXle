(define (domain test-negative)
  (:requirements :strips :typing :negative-preconditions)
  (:types switch)
  (:predicates
    (on ?s - switch)
  )
  (:action toggle-on
    :parameters (?s - switch)
    :precondition (not (on ?s))
    :effect (on ?s)
  )
  (:action toggle-off
    :parameters (?s - switch)
    :precondition (on ?s)
    :effect (not (on ?s))
  )
)
