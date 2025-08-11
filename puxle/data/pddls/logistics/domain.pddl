(define (domain logistics)
  (:requirements :strips :typing)
  (:types city location vehicle package)
  (:predicates
    (in-city ?l - location ?c - city)
    (road ?from - location ?to - location)
    (air ?from - location ?to - location)
    (is-truck ?v - vehicle)
    (is-plane ?v - vehicle)
    (at-veh ?v - vehicle ?l - location)
    (at-pkg ?p - package ?l - location)
    (in ?p - package ?v - vehicle)
  )

  (:action load
    :parameters (?p - package ?v - vehicle ?l - location)
    :precondition (and (at-pkg ?p ?l) (at-veh ?v ?l))
    :effect (and (in ?p ?v) (not (at-pkg ?p ?l)))
  )

  (:action unload
    :parameters (?p - package ?v - vehicle ?l - location)
    :precondition (and (in ?p ?v) (at-veh ?v ?l))
    :effect (and (at-pkg ?p ?l) (not (in ?p ?v)))
  )

  (:action drive
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (is-truck ?v) (at-veh ?v ?from) (road ?from ?to))
    :effect (and (at-veh ?v ?to) (not (at-veh ?v ?from)))
  )

  (:action fly
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (is-plane ?v) (at-veh ?v ?from) (air ?from ?to))
    :effect (and (at-veh ?v ?to) (not (at-veh ?v ?from)))
  )
)
