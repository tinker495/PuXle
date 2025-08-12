(define (problem rv-H-01)
  (:domain rovers)
  (:objects r1 r2 - rover
            w1 w2 w3 w4 w5 w6 - waypoint
            i1 i2 - instrument
            s1 s2 s3 s4 - sample
            o1 o2 o3 o4 - objective)
  (:init
    (at r1 w1)
    (at r2 w4)
    (adj w1 w2)
    (adj w2 w3)
    (adj w3 w4)
    (adj w4 w5)
    (adj w5 w6)
    (adj w6 w1)
    (adj w2 w1)
    (adj w3 w2)
    (adj w4 w3)
    (adj w5 w4)
    (adj w6 w5)

    (have-instrument r1 i1)
    (have-instrument r2 i2)

    (at-sample s1 w2)
    (at-sample s2 w3)
    (at-sample s3 w5)
    (at-sample s4 w6)

    (visible w2 o1)
    (visible w3 o2)
    (visible w5 o3)
    (visible w6 o4)
  )
  (:goal (and
    (communicated-image o1)
    (communicated-image o2)
    (communicated-image o3)
    (communicated-image o4)
    (communicated-sample s1)
    (communicated-sample s2)
    (communicated-sample s3)
    (communicated-sample s4)
  ))
)
