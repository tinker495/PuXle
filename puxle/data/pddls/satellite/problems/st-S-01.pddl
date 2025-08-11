(define (problem st-S-01)
  (:domain satellite)
  (:objects s1 - satellite
            i1 - instrument
            m1 - mode
            d1 d2 - direction
            t1 - target)
  (:init
    (on-board i1 s1)
    (supports i1 m1)
    (pointing s1 d1)
    (calibration-target i1 d1)
  )
  (:goal (and (have-image t1 m1)))
)
