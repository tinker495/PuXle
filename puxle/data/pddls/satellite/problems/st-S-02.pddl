(define (problem st-S-02)
  (:domain satellite)
  (:objects s1 - satellite
            i1 i2 - instrument
            m1 m2 - mode
            d1 d2 - direction
            t1 t2 - target)
  (:init
    (on-board i1 s1)
    (on-board i2 s1)
    (supports i1 m1)
    (supports i2 m2)
    (pointing s1 d1)
    (calibration-target i1 d1)
    (calibration-target i2 d2)
  )
  (:goal (and (have-image t1 m1) (have-image t2 m2)))
)
