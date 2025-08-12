(define (problem st-S-04)
  (:domain satellite)
  (:objects s1 - satellite
            i1 i2 - instrument
            m1 m2 - mode
            d1 d2 d3 d4 - direction
            t1 t2 t3 t4 - target)
  (:init
    (on-board i1 s1)
    (on-board i2 s1)
    (supports i1 m1)
    (supports i2 m2)
    (pointing s1 d1)
    (calibration-target i1 d1)
    (calibration-target i2 d3)
  )
  (:goal (and
    (have-image t1 m1)
    (have-image t2 m2)
    (have-image t3 m1)
    (have-image t4 m2)
  ))
)
