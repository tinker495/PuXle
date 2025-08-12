(define (problem lg-H-01)
  (:domain logistics)
  (:objects
    c1 c2 c3 - city
    l1 l2 l3 - location
    a1 a2 a3 - location
    t1 t2 - vehicle
    p1 p2 - vehicle
    pkg1 pkg2 pkg3 pkg4 pkg5 pkg6 - package
  )
  (:init
    (in-city l1 c1)
    (in-city l2 c1)
    (in-city l3 c1)
    (in-city a1 c2)
    (in-city a2 c2)
    (in-city a3 c2)
    (in-city a1 c2)
    (in-city a2 c2)
    (in-city a3 c2)
    (in-city l1 c1)
    (in-city l2 c1)
    (in-city l3 c1)

    (road l1 l2)
    (road l2 l3)
    (road l3 l1)
    (road a1 a2)
    (road a2 a3)
    (road a3 a1)

    (air l1 a1)
    (air l2 a2)
    (air l3 a3)
    (air a1 l2)
    (air a2 l3)
    (air a3 l1)

    (is-truck t1)
    (is-truck t2)
    (is-plane p1)
    (is-plane p2)

    (at-veh t1 l1)
    (at-veh t2 l2)
    (at-veh p1 a1)
    (at-veh p2 a2)

    (at-pkg pkg1 l3)
    (at-pkg pkg2 l2)
    (at-pkg pkg3 l1)
    (at-pkg pkg4 a1)
    (at-pkg pkg5 a2)
    (at-pkg pkg6 a3)
  )
  (:goal (and
    (at-pkg pkg1 a1)
    (at-pkg pkg2 a2)
    (at-pkg pkg3 a3)
    (at-pkg pkg4 l1)
    (at-pkg pkg5 l2)
    (at-pkg pkg6 l3)
  ))
)
