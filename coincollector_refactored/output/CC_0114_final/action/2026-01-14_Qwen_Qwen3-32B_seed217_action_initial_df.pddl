(define (domain exploration)
  (:requirements :strips :typing)
  (:types location direction)
  (:predicates
    (at ?l - location)
    (open ?from - location ?to - location ?d - direction)
    (closed ?from - location ?to - location ?d - direction)
  )
  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (closed ?loc1 ?loc2 ?dir))
    :effect (and (open ?loc1 ?loc2 ?dir) (not (closed ?loc1 ?loc2 ?dir)))
  )
  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (open ?from ?to ?dir))
    :effect (and (at ?to) (not (at ?from)))
  )
)