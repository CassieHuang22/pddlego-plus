(define (domain house-exploration)
  (:requirements :strips :typing)
  (:types location direction)

  (:predicates
    (at ?loc - location)
    (door ?loc1 - location ?loc2 - location ?dir - direction)
    (door-open ?loc1 - location ?loc2 - location)
    (door-closed ?loc1 - location ?loc2 - location)
  )

  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (door ?loc1 ?loc2 ?dir) (door-closed ?loc1 ?loc2))
    :effect (and (door-open ?loc1 ?loc2) (not (door-closed ?loc1 ?loc2)))
  )
  
  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (door ?from ?to ?dir) (door-open ?from ?to))
    :effect (and (at ?to) (not (at ?from)))
  )
)
