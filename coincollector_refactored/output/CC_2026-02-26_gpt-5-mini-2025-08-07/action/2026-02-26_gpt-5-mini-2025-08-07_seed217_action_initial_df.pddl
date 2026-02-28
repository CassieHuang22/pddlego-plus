(define (domain house-exploration)
  (:requirements :typing :strips)
  (:types location direction)
  (:predicates
    (at ?l - location)
    (door-closed ?l1 - location ?l2 - location ?d - direction)
    (door-open ?l1 - location ?l2 - location ?d - direction)
    (adjacent ?l1 - location ?l2 - location ?d - direction)
  )

  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (adjacent ?loc1 ?loc2 ?dir) (door-closed ?loc1 ?loc2 ?dir))
    :effect (and (not (door-closed ?loc1 ?loc2 ?dir)) (door-open ?loc1 ?loc2 ?dir) (not (at ?loc1)) (at ?loc2))
  )

  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (adjacent ?from ?to ?dir) (door-open ?from ?to ?dir))
    :effect (and (not (at ?from)) (at ?to))
  )
)
