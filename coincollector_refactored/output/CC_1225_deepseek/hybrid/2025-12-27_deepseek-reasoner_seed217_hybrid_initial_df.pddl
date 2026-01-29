(define (domain exploration-domain)
  (:requirements :strips)
  (:types location direction)
  (:predicates 
    (at ?l - location)
    (connected ?l1 - location ?l2 - location ?d - direction)
    (closed ?l1 - location ?l2 - location ?d - direction)
  )
  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (closed ?loc1 ?loc2 ?dir))
    :effect (and (not (closed ?loc1 ?loc2 ?dir)) (connected ?loc1 ?loc2 ?dir))
  )
  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (connected ?from ?to ?dir))
    :effect (and (not (at ?from)) (at ?to))
  )
)