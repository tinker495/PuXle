(define (domain rovers)
  (:requirements :strips :typing)
  (:types rover waypoint sample
          objective instrument)
  (:predicates
    (at ?r - rover ?w - waypoint)
    (adj ?w1 - waypoint ?w2 - waypoint)
    (calibrated ?r - rover ?i - instrument)
    (have-instrument ?r - rover ?i - instrument)
    (have-sample ?r - rover ?s - sample)
    (at-sample ?s - sample ?w - waypoint)
    (visible ?w - waypoint ?o - objective)
    (have-image ?o - objective)
    (communicated-image ?o - objective)
    (communicated-sample ?s - sample)
  )

  (:action navigate
    :parameters (?r - rover ?from - waypoint ?to - waypoint)
    :precondition (and (at ?r ?from) (adj ?from ?to))
    :effect (and (at ?r ?to) (not (at ?r ?from)))
  )

  (:action calibrate
    :parameters (?r - rover ?i - instrument)
    :precondition (have-instrument ?r ?i)
    :effect (calibrated ?r ?i)
  )

  (:action take-image
    :parameters (?r - rover ?w - waypoint ?o - objective ?i - instrument)
    :precondition (and (at ?r ?w) (have-instrument ?r ?i) (calibrated ?r ?i) (visible ?w ?o))
    :effect (have-image ?o)
  )

  (:action sample-rock
    :parameters (?r - rover ?s - sample ?w - waypoint)
    :precondition (and (at ?r ?w) (at-sample ?s ?w))
    :effect (have-sample ?r ?s)
  )

  (:action communicate-image
    :parameters (?r - rover ?o - objective)
    :precondition (have-image ?o)
    :effect (communicated-image ?o)
  )

  (:action communicate-sample
    :parameters (?r - rover ?s - sample)
    :precondition (have-sample ?r ?s)
    :effect (communicated-sample ?s)
  )
)
