(define (problem rv-S-01)
  (:domain rovers)
  (:objects r1 - rover
            w1 w2 - waypoint
            i1 - instrument
            s1 - sample
            o1 - objective)
  (:init
    (at r1 w1)
    (adj w1 w2)
    (adj w2 w1)
    (have-instrument r1 i1)
    (at-sample s1 w2)
    (visible w2 o1)
  )
  (:goal (and (communicated-image o1) (communicated-sample s1)))
)
