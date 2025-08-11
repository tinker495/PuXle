(define (domain satellite)
  (:requirements :strips :typing)
  (:types satellite instrument mode direction target)
  (:predicates
    (on-board ?i - instrument ?s - satellite)
    (supports ?i - instrument ?m - mode)
    (calibrated ?i - instrument)
    (pointing ?s - satellite ?d - direction)
    (calibration-target ?i - instrument ?d - direction)
    (have-image ?t - target ?m - mode)
  )

  (:action turn-to
    :parameters (?s - satellite ?from - direction ?to - direction)
    :precondition (pointing ?s ?from)
    :effect (and (pointing ?s ?to) (not (pointing ?s ?from)))
  )

  (:action calibrate
    :parameters (?s - satellite ?i - instrument ?d - direction)
    :precondition (and (on-board ?i ?s) (pointing ?s ?d) (calibration-target ?i ?d))
    :effect (calibrated ?i)
  )

  (:action take-image
    :parameters (?s - satellite ?i - instrument ?d - direction ?t - target ?m - mode)
    :precondition (and (on-board ?i ?s) (pointing ?s ?d) (supports ?i ?m) (calibrated ?i))
    :effect (have-image ?t ?m)
  )
)
