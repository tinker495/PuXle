(define (problem lg-S-01)
  (:domain logistics)
  (:objects
    c1 c2 - city
    l1 l2 - location
    a1 a2 - location
    t1 - vehicle
    p1 - vehicle
    pkg1 pkg2 - package
  )
  (:init
    (in-city l1 c1)
    (in-city l2 c1)
    (in-city a1 c2)
    (in-city a2 c2)

    (road l1 l2)
    (road l2 l1)
    (road a1 a2)
    (road a2 a1)

    (air l1 a1)
    (air a1 l1)
    (air l2 a2)
    (air a2 l2)

    ; cross-city air links to enable inter-city transport
    (air l1 a2)
    (air a2 l1)
    (air l2 a1)
    (air a1 l2)

    (is-truck t1)
    (is-plane p1)

    (at-veh t1 l1)
    (at-veh p1 a1)

    (at-pkg pkg1 l2)
    (at-pkg pkg2 l1)
  )
  (:goal (and (at-pkg pkg1 a2) (at-pkg pkg2 a1)))
)
