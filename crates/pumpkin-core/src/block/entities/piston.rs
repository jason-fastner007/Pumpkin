use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam::atomic::AtomicCell;
use pumpkin_data::{Block, BlockDirection, BlockState, entity::EntityType};
use pumpkin_nbt::compound::NbtCompound;
use pumpkin_util::math::{boundingbox::BoundingBox, position::BlockPos, vector3::Vector3};

use crate::entity::{EntityBase, decoration::armor_stand::ArmorStandEntity};
use crate::world::{BlockFlags, World};

use super::BlockEntity;

pub struct PistonBlockEntity {
    pub position: BlockPos,
    pub pushed_block_state: &'static BlockState,
    pub facing: BlockDirection,
    pub current_progress: AtomicCell<f32>,
    pub last_progress: AtomicCell<f32>,
    pub extending: bool,
    pub source: bool,
}

impl PistonBlockEntity {
    pub const ID: &'static str = "minecraft:piston";

    const fn movement_direction(&self) -> BlockDirection {
        if self.extending {
            self.facing
        } else {
            self.facing.opposite()
        }
    }

    /// Vanilla's `getAmountExtended`: how far back from the block's final position
    /// the visual is at a given animation progress. Negative for extending.
    fn amount_extended(&self, progress: f32) -> f32 {
        if self.extending {
            progress - 1.0
        } else {
            1.0 - progress
        }
    }

    fn dir_vec(dir: BlockDirection, scale: f64) -> Vector3<f64> {
        let off = dir.to_offset();
        Vector3::new(
            f64::from(off.x) * scale,
            f64::from(off.y) * scale,
            f64::from(off.z) * scale,
        )
    }

    /// Ports vanilla `PistonBlockEntity.pushEntities`: pushes entities whose
    /// bounding box intersects the moving block's swept volume during this tick.
    fn push_entities(&self, world: &Arc<World>, new_progress: f32) {
        let last = self.last_progress.load();
        let delta = f64::from(new_progress - last);
        if delta <= 0.0 {
            return;
        }

        let motion_dir = self.movement_direction();
        let amount = f64::from(self.amount_extended(last));

        // Block AABB at the current visual position (start of this tick).
        // For source=true (the piston head BE), vanilla uses the piston-head
        // collision shape — a 4-pixel-thick slab at the front face — rather than
        // the full cube. Without this, the head "sweeps" too much volume and
        // drags entities (e.g. end crystals) back with it during retraction.
        let raw_block_aabb = BoundingBox::from_block(&self.position);
        let block_aabb = if self.source {
            Self::head_shape_aabb(raw_block_aabb, self.facing)
        } else {
            raw_block_aabb
        }
        .shift(Self::dir_vec(self.facing, amount));

        // Stretch by motion to get the swept volume (one tick of animation).
        let motion = Self::dir_vec(motion_dir, delta);
        let swept = block_aabb.stretch(motion);
        // Vanilla `PistonMath.getMovementArea`: only the slab the moving face sweeps this tick.
        // Entities are pushed only when they are in it, never when they sit on the trailing side,
        // such as one standing on a block that retracts downwards.
        let movement_area = Self::movement_area(block_aabb, motion_dir, delta);
        let slime = Block::from_state_id(self.pushed_block_state.id) == &Block::SLIME_BLOCK;

        for entity in world.get_entities_at_box(&swept) {
            let e = entity.get_entity();
            if e.no_physics.load(Ordering::Relaxed) {
                continue;
            }
            // Player movement is client-authoritative; vanilla still nudges them
            // via Entity.move(PISTON), but we skip them here to avoid teleport jank.
            // Vanilla also skips `ServerPlayer` for the slime launch: the client launches its own.
            if entity.get_player().is_some() || Self::ignores_pistons(entity.as_ref()) {
                continue;
            }

            // Only entities in front of the moving face are launched or pushed. Vanilla launches
            // everything touching the block, but there the moving block still collides, so nothing
            // sinks into it from behind. Pumpkin's moving piston has no collision yet: an entity
            // resting on a slime block that retracts downwards sinks in for a tick, and launching
            // it then threw it down through the floor.
            let entity_aabb = e.bounding_box.load();
            if !movement_area.intersects(&entity_aabb) {
                continue;
            }

            // Vanilla `moveCollidedEntities`: a moving slime block sets the entity's velocity
            // along the motion axis to the motion direction (+-1), then pushes it like any block.
            if slime {
                Self::launch(e, motion_dir);
            }
            let intersection = Self::intersection_size(movement_area, motion_dir, entity_aabb);
            if intersection <= 0.0 {
                continue;
            }
            let push_amount = intersection.min(delta) + 0.01;
            Self::move_entity(e, motion_dir, push_amount);

            // For a retracting head, vanilla also shoves the entity OUT of the
            // piston body cube. Without this, entities get pulled into the
            // piston and "stick" to it (looks like a sticky-piston drag).
            // For a retract-head BE, `self.position` already IS the piston
            // block position (it replaces the piston during animation).
            if !self.extending && self.source {
                Self::push_out_of_piston_body(e, &self.position, motion_dir, delta);
            }
        }
    }

    /// Vanilla `PistonMath.getMovementArea`: the part of `aabb`'s leading face swept by moving
    /// `distance` towards `dir`.
    fn movement_area(aabb: BoundingBox, dir: BlockDirection, distance: f64) -> BoundingBox {
        let (min, max) = (aabb.min, aabb.max);
        let (lo, hi) = match dir {
            BlockDirection::Down => (
                Vector3::new(min.x, min.y - distance, min.z),
                Vector3::new(max.x, min.y, max.z),
            ),
            BlockDirection::Up => (
                Vector3::new(min.x, max.y, min.z),
                Vector3::new(max.x, max.y + distance, max.z),
            ),
            BlockDirection::North => (
                Vector3::new(min.x, min.y, min.z - distance),
                Vector3::new(max.x, max.y, min.z),
            ),
            BlockDirection::South => (
                Vector3::new(min.x, min.y, max.z),
                Vector3::new(max.x, max.y, max.z + distance),
            ),
            BlockDirection::West => (
                Vector3::new(min.x - distance, min.y, min.z),
                Vector3::new(min.x, max.y, max.z),
            ),
            BlockDirection::East => (
                Vector3::new(max.x, min.y, min.z),
                Vector3::new(max.x + distance, max.y, max.z),
            ),
        };
        BoundingBox::new(lo, hi)
    }

    /// Vanilla `getIntersectionSize`: how much `entity` overlaps `swept` along
    /// `motion_dir`. Positive means the entity is in the path of the moving block.
    fn intersection_size(
        swept: BoundingBox,
        motion_dir: BlockDirection,
        entity: BoundingBox,
    ) -> f64 {
        match motion_dir {
            BlockDirection::East => swept.max.x - entity.min.x,
            BlockDirection::West => entity.max.x - swept.min.x,
            BlockDirection::Up => swept.max.y - entity.min.y,
            BlockDirection::Down => entity.max.y - swept.min.y,
            BlockDirection::South => swept.max.z - entity.min.z,
            BlockDirection::North => entity.max.z - swept.min.z,
        }
    }

    /// Entities whose vanilla `getPistonPushReaction` is `IGNORE`: pistons neither push nor
    /// launch them.
    fn ignores_pistons(entity: &dyn EntityBase) -> bool {
        let entity_type = entity.get_entity().entity_type;
        [
            &EntityType::AREA_EFFECT_CLOUD,
            &EntityType::MARKER,
            &EntityType::BLOCK_DISPLAY,
            &EntityType::ITEM_DISPLAY,
            &EntityType::TEXT_DISPLAY,
            &EntityType::INTERACTION,
            &EntityType::PAINTING,
            &EntityType::ITEM_FRAME,
            &EntityType::GLOW_ITEM_FRAME,
            &EntityType::LEASH_KNOT,
        ]
        .contains(&entity_type)
            || entity
                .cast_any()
                .downcast_ref::<ArmorStandEntity>()
                .is_some_and(ArmorStandEntity::is_marker)
    }

    /// Vanilla slime launch: the velocity component along `dir`'s axis becomes `dir`'s step.
    fn launch(entity: &crate::entity::Entity, dir: BlockDirection) {
        let off = dir.to_offset();
        let v = entity.velocity.load();
        entity.set_velocity(Vector3::new(
            if off.x == 0 { v.x } else { f64::from(off.x) },
            if off.y == 0 { v.y } else { f64::from(off.y) },
            if off.z == 0 { v.z } else { f64::from(off.z) },
        ));
    }

    fn move_entity(entity: &crate::entity::Entity, dir: BlockDirection, distance: f64) {
        let new_pos = entity.pos.load() + Self::dir_vec(dir, distance);
        entity.set_pos(new_pos);
    }

    /// Vanilla `push`: when a piston head retracts, shove entities that ended up
    /// inside the piston-body cube back out the opposite direction (slightly past
    /// the move they just got, so the net motion is essentially zero).
    fn push_out_of_piston_body(
        entity: &crate::entity::Entity,
        piston_pos: &BlockPos,
        motion_dir: BlockDirection,
        amount: f64,
    ) {
        let body_aabb = BoundingBox::from_block(piston_pos);
        let entity_aabb = entity.bounding_box.load();
        if !body_aabb.intersects(&entity_aabb) {
            return;
        }
        let back = motion_dir.opposite();
        let e = Self::intersection_size(body_aabb, back, entity_aabb) + 0.01;
        let f = Self::intersection_size(
            body_aabb,
            back,
            Self::aabb_intersection(body_aabb, entity_aabb),
        ) + 0.01;
        if (e - f).abs() < 0.01 {
            let distance = e.min(amount) + 0.01;
            Self::move_entity(entity, back, distance);
        }
    }

    /// Approximation of vanilla `PistonHeadBlock` collision shape: a 4-pixel
    /// (0.25 block) slab at the front face of the block in the facing direction.
    fn head_shape_aabb(block_aabb: BoundingBox, facing: BlockDirection) -> BoundingBox {
        const HEAD_THICKNESS: f64 = 0.25;
        let mut min = block_aabb.min;
        let mut max = block_aabb.max;
        match facing {
            BlockDirection::East => min.x = max.x - HEAD_THICKNESS,
            BlockDirection::West => max.x = min.x + HEAD_THICKNESS,
            BlockDirection::Up => min.y = max.y - HEAD_THICKNESS,
            BlockDirection::Down => max.y = min.y + HEAD_THICKNESS,
            BlockDirection::South => min.z = max.z - HEAD_THICKNESS,
            BlockDirection::North => max.z = min.z + HEAD_THICKNESS,
        }
        BoundingBox::new(min, max)
    }

    const fn aabb_intersection(a: BoundingBox, b: BoundingBox) -> BoundingBox {
        BoundingBox::new(
            Vector3::new(
                a.min.x.max(b.min.x),
                a.min.y.max(b.min.y),
                a.min.z.max(b.min.z),
            ),
            Vector3::new(
                a.max.x.min(b.max.x),
                a.max.y.min(b.max.y),
                a.max.z.min(b.max.z),
            ),
        )
    }

    pub fn finish(&self, world: &Arc<World>) {
        if self.last_progress.load() < 1.0 {
            let pos = self.position;
            world.remove_block_entity(&pos);
            if world.get_block(&pos) == &Block::MOVING_PISTON {
                let state = if self.source {
                    Block::AIR.default_state.id
                } else {
                    world.update_from_neighbor_shapes(self.pushed_block_state.id, &pos)
                };
                world.set_block_state(&pos, state, BlockFlags::NOTIFY_ALL);
                world.update_neighbors(&pos, None);
            }
        }
    }
}

const FACING: &str = "facing";
const LAST_PROGRESS: &str = "progress";
const EXTENDING: &str = "extending";
const SOURCE: &str = "source";
const BLOCK_STATE: &str = "blockState";

/// A block state as vanilla writes it in NBT: `Name` plus `Properties` as strings.
fn block_state_nbt(state: &BlockState) -> NbtCompound {
    let block = Block::from_state_id(state.id);
    let mut nbt = NbtCompound::new();
    let name = if block.name.starts_with("minecraft:") {
        block.name.to_string()
    } else {
        format!("minecraft:{}", block.name)
    };
    nbt.put_string("Name", name);
    if let Some(props) = block.properties(state.id) {
        let props = props.to_props();
        if !props.is_empty() {
            let mut props_nbt = NbtCompound::new();
            for prop in props {
                props_nbt.put_string(prop.key_str(), prop.value_str().to_string());
            }
            nbt.put_compound("Properties", props_nbt);
        }
    }
    nbt
}

impl BlockEntity for PistonBlockEntity {
    fn resource_location(&self) -> &'static str {
        Self::ID
    }

    fn get_position(&self) -> BlockPos {
        self.position
    }

    fn tick(&self, world: &Arc<World>) {
        let current_progress = self.current_progress.load();
        self.last_progress.store(current_progress);
        if current_progress >= 1.0 {
            let pos = self.position;
            world.remove_block_entity(&pos);
            if world.get_block(&pos) == &Block::MOVING_PISTON {
                if self.pushed_block_state.is_air() {
                    world.set_block_state(
                        &pos,
                        self.pushed_block_state.id,
                        BlockFlags::FORCE_STATE | BlockFlags::MOVED,
                    );
                } else {
                    let updated_state =
                        world.update_from_neighbor_shapes(self.pushed_block_state.id, &pos);
                    world.set_block_state(
                        &pos,
                        updated_state,
                        BlockFlags::NOTIFY_ALL | BlockFlags::MOVED,
                    );
                    world.update_neighbors(&pos, None);
                }
            }
            return;
        }
        let new_progress = (current_progress + 0.5).min(1.0);
        self.push_entities(world, new_progress);
        self.current_progress.store(new_progress);
    }

    fn from_nbt(nbt: &pumpkin_nbt::compound::NbtCompound, position: BlockPos) -> Self
    where
        Self: Sized,
    {
        // TODO
        let pushed_block_state = Block::AIR.default_state;
        let facing = nbt.get_byte(FACING).unwrap_or(0);
        let last_progress = nbt.get_float(LAST_PROGRESS).unwrap_or(0.0);
        let extending = nbt.get_bool(EXTENDING).unwrap_or(false);
        let source = nbt.get_bool(SOURCE).unwrap_or(false);
        Self {
            pushed_block_state,
            position,
            facing: BlockDirection::from_index(facing as u8).unwrap_or(BlockDirection::Down),
            current_progress: last_progress.into(),
            last_progress: last_progress.into(),
            extending,
            source,
        }
    }

    fn write_nbt(&self, nbt: &mut NbtCompound) {
        // TODO: pushed_block_state
        nbt.put_byte(FACING, self.facing.to_index() as i8);
        nbt.put_float(LAST_PROGRESS, self.last_progress.load());
        nbt.put_bool(EXTENDING, self.extending);
        nbt.put_bool(SOURCE, self.source);
    }

    fn chunk_data_nbt(&self) -> Option<NbtCompound> {
        // What a client gets when it loads the chunk mid-animation, as in vanilla
        // `PistonMovingBlockEntity.getUpdateTag`. Without `blockState` the client would animate
        // air: no collision, so a slime block would not launch the player.
        let mut nbt = NbtCompound::new();
        nbt.put_compound(BLOCK_STATE, block_state_nbt(self.pushed_block_state));
        nbt.put_int(FACING, i32::from(self.facing.to_index()));
        nbt.put_float(LAST_PROGRESS, self.last_progress.load());
        nbt.put_bool(EXTENDING, self.extending);
        nbt.put_bool(SOURCE, self.source);
        Some(nbt)
    }

    fn update_packet_nbt(&self) -> Option<NbtCompound> {
        // Vanilla sends no update packet for a moving piston: the client animates it on its own
        // from the block event, and launches its player off a slime block from that copy. An
        // update would reload the client's copy mid-animation.
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use pumpkin_data::BlockDirection;
    use pumpkin_util::math::{boundingbox::BoundingBox, position::BlockPos, vector3::Vector3};

    use super::PistonBlockEntity;

    fn block() -> BoundingBox {
        BoundingBox::from_block(&BlockPos::new(0, 0, 0))
    }

    #[test]
    fn movement_area_is_the_slab_in_front_of_the_moving_face() {
        let cases = [
            (BlockDirection::Up, [0.0, 1.0, 0.0], [1.0, 1.5, 1.0]),
            (BlockDirection::Down, [0.0, -0.5, 0.0], [1.0, 0.0, 1.0]),
            (BlockDirection::North, [0.0, 0.0, -0.5], [1.0, 1.0, 0.0]),
            (BlockDirection::South, [0.0, 0.0, 1.0], [1.0, 1.0, 1.5]),
            (BlockDirection::West, [-0.5, 0.0, 0.0], [0.0, 1.0, 1.0]),
            (BlockDirection::East, [1.0, 0.0, 0.0], [1.5, 1.0, 1.0]),
        ];
        for (dir, min, max) in cases {
            let area = PistonBlockEntity::movement_area(block(), dir, 0.5);
            assert_eq!(area.min, Vector3::new(min[0], min[1], min[2]), "{dir:?}");
            assert_eq!(area.max, Vector3::new(max[0], max[1], max[2]), "{dir:?}");
        }
    }

    #[test]
    fn entity_on_top_of_a_block_moving_down_is_not_pushed() {
        // An armor-stand-sized entity resting on the block, slightly sunk in as after landing.
        let on_top = BoundingBox::new(
            Vector3::new(0.25, 0.99, 0.25),
            Vector3::new(0.75, 2.97, 0.75),
        );
        let down = PistonBlockEntity::movement_area(block(), BlockDirection::Down, 0.5);
        assert!(!down.intersects(&on_top));

        // The same entity is in the way when the block moves up.
        let up = PistonBlockEntity::movement_area(block(), BlockDirection::Up, 0.5);
        assert!(up.intersects(&on_top));
    }
}
