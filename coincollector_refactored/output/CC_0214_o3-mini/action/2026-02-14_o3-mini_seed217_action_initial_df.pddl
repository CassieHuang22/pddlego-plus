(define (domain exploration)
  (:requirements :strips :typing)
  (:types location direction)
  (:predicates
    (at ?loc - location)
    (door_closed ?loc1 - location ?loc2 - location ?dir - direction)
    (door_open ?loc1 - location ?loc2 - location ?dir - direction)
  )

  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (door_closed ?loc1 ?loc2 ?dir))
    :effect (and (not (door_closed ?loc1 ?loc2 ?dir))
                 (door_open ?loc1 ?loc2 ?dir)
                 (not (at ?loc1))
                 (at ?loc2)))

  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (door_open ?from ?to ?dir))
    :effect (and (not (at ?from))
                 (at ?to)))
)