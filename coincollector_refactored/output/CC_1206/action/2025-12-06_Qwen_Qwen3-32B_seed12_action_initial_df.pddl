(define (domain exploration)
    (:requirements :strips :typing)
    (:types location direction - object)
    (:predicates
        (at ?loc - location)
        (closed ?from - location ?to - location ?dir - direction)
        (open ?from - location ?to - location ?dir - direction)
    )
    (:action open-door
        :parameters (?loc1 - location ?loc2 - location ?dir - direction)
        :precondition (and (closed ?loc1 ?loc2 ?dir) (at ?loc1))
        :effect (and (not (closed ?loc1 ?loc2 ?dir)) (open ?loc1 ?loc2 ?dir))
    )
    (:action move
        :parameters (?from - location ?to - location ?dir - direction)
        :precondition (and (open ?from ?to ?dir) (at ?from))
        :effect (and (not (at ?from)) (at ?to))
    )
)