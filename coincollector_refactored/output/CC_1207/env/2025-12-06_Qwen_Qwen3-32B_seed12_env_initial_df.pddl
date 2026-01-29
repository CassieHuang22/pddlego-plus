(define (domain exploration)
  (:predicates
    (at ?l - location)
    (closed ?to - location ?from - location ?dir - direction)
    (open ?to - location ?from - location ?dir - direction)
    (connected ?from - location ?to - location ?dir - direction)
  )
  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (closed ?loc2 ?loc1 ?dir) (connected ?loc1 ?loc2 ?dir))
    :effect (and (open ?loc2 ?loc1 ?dir) (not (closed ?loc2 ?loc1 ?dir)))
  )
  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (open ?to ?from ?dir) (connected ?from ?to ?dir))
    :effect (and (at ?to) (not (at ?from)))
  )
)
