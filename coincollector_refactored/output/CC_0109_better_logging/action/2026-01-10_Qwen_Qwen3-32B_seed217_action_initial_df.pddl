(define (domain exploration)
  (:requirements :strips :typing)
  (:types location direction)
  (:predicates
    (at ?l - location)
    (door-closed ?from - location ?to - location ?dir - direction)
    (door-open ?from - location ?to - location ?dir - direction)
  )

  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (door-closed ?loc1 ?loc2 ?dir)
    :effect (and (not (door-closed ?loc1 ?loc2 ?dir)) (door-open ?loc1 ?loc2 ?dir))
  )

  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (door-open ?from ?to ?dir))
    :effect (and (not (at ?from)) (at ?to))
  )
)