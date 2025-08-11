(define (problem rv-S-02)
  (:domain rovers)
  (:objects r1 - rover
            w1 w2 w3 - waypoint
            i1 - instrument
            s1 s2 - sample
            o1 o2 - objective)
  (:init
    (at r1 w1)
    (adj w1 w2)
    (adj w2 w3)
    (adj w3 w2)
    (adj w2 w1)
    (have-instrument r1 i1)
    (at-sample s1 w2)
    (at-sample s2 w3)
    (visible w2 o1)
    (visible w3 o2)
  )
  (:goal (and (communicated-image o1) (communicated-sample s1) (communicated-image o2)))
)
