(define (problem unreachable-move-problem)
  (:domain simple-move)
  (:objects loc1 loc2 loc3 - location)
  (:init
    (at loc1)
    (connected loc1 loc2)
    ; Note: no path from loc2 to loc3 and no loc2<-loc1 return
  )
  (:goal (at loc3))
)
