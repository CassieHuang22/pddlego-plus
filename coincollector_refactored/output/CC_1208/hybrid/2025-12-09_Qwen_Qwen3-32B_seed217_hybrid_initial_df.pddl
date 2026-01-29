(define (domain exploration)
  (:types location direction)
  (:predicates
    (at ?loc - location)
    (connected ?loc1 - location ?loc2 - location ?dir - direction)
    (closed ?loc1 - location ?loc2 - location ?dir - direction)
    (opposite ?dir1 - direction ?dir2 - direction)
  )
  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (closed ?loc1 ?loc2 ?dir))
    :effect (and 
        (not (closed ?loc1 ?loc2 ?dir))
        (forall (?d - direction) (when (opposite ?dir ?d) (not (closed ?loc2 ?loc1 ?d))))
    )
  )
  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (connected ?from ?to ?dir) (not (closed ?from ?to ?dir)))
    :effect (and (not (at ?from)) (at ?to))
  )
)