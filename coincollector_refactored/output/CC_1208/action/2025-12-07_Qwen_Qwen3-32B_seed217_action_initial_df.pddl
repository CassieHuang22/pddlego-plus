(define (domain exploration)
  (:requirements :strips :typing)
  (:types location direction - object)
  (:predicates
    (closed ?loc - location ?dir - direction)
    (connected ?loc1 - location ?loc2 - location ?dir - direction)
    (at ?loc - location)
  )
  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (closed ?loc1 ?dir))
    :effect (and (not (closed ?loc1 ?dir)) (connected ?loc1 ?loc2 ?dir))
  )
  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (connected ?from ?to ?dir))
    :effect (and (not (at ?from)) (at ?to))
  )
)