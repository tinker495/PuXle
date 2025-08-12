(define (problem gr-S-04)
  (:domain gripper)
  (:objects room1 room2 - room
            ball1 ball2 ball3 ball4 ball5 - ball
            left right - gripper)
  (:init
    (at-robby room1)
    (at ball1 room1)
    (at ball2 room1)
    (at ball3 room1)
    (at ball4 room1)
    (at ball5 room1)
    (free left)
    (free right)
  )
  (:goal (and
    (at ball1 room2)
    (at ball2 room2)
    (at ball3 room2)
    (at ball4 room2)
    (at ball5 room2)
  ))
)
