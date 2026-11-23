#!/usr/bin/env python3
"""Unit tests for the new MSG_QR_RESET (0x06) protocol and state reset handling."""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upper_machine.comm.codec import RdlcCodec
from upper_machine.comm.frame import RdlcFrame
from upper_machine.serial_comm import (
    SerialSender,
    SerialRxEvents,
    MSG_MODE_CMD,
    MSG_GRIPPER_COUNT,
    MSG_QR_RESET,
    MODE_BLOCK,
    MODE_RING,
    GRIPPER_PICK_INC,
)
from upper_machine.status_display import TaskDisplay


class TestQrResetProtocol(unittest.TestCase):
    def setUp(self):
        self.codec = RdlcCodec()

    def test_decode_qr_reset_2bytes(self):
        """Test decoding 2-byte QR reset frame: [0x06, 0x01]."""
        sender = SerialSender(port="dummy", baudrate=115200)
        sender.link = MagicMock()
        
        # Mock receive queue with 1 frame
        frame = RdlcFrame(src=0x02, dst=0x01, payload=bytes([MSG_QR_RESET, 0x01]))
        from queue import Empty
        sender.link.receive.side_effect = [frame, Empty()]

        events = sender.poll_events()
        self.assertTrue(events.qr_reset)
        self.assertIsNone(events.new_mode)
        self.assertEqual(events.pick_delta, 0)

    def test_decode_qr_reset_1byte_tolerant(self):
        """Test decoding 1-byte QR reset frame: [0x06]."""
        sender = SerialSender(port="dummy", baudrate=115200)
        sender.link = MagicMock()
        
        frame = RdlcFrame(src=0x02, dst=0x01, payload=bytes([MSG_QR_RESET]))
        from queue import Empty
        sender.link.receive.side_effect = [frame, Empty()]

        events = sender.poll_events()
        self.assertTrue(events.qr_reset)

    def test_existing_protocols_regression(self):
        """Ensure MSG_MODE_CMD (0x04) and MSG_GRIPPER_COUNT (0x05) still decode properly."""
        sender = SerialSender(port="dummy", baudrate=115200)
        sender.link = MagicMock()

        frame_mode = RdlcFrame(src=0x02, dst=0x01, payload=bytes([MSG_MODE_CMD, MODE_RING]))
        frame_gripper = RdlcFrame(src=0x02, dst=0x01, payload=bytes([MSG_GRIPPER_COUNT, GRIPPER_PICK_INC]))
        from queue import Empty
        sender.link.receive.side_effect = [frame_mode, frame_gripper, Empty()]

        events = sender.poll_events()
        self.assertEqual(events.new_mode, MODE_RING)
        self.assertEqual(events.pick_delta, 1)
        self.assertFalse(events.qr_reset)

    def test_task_display_reset(self):
        """Test TaskDisplay task_code reset behavior."""
        display = TaskDisplay(width=640, height=480, fullscreen=False)
        display.set_task_code("123+456+123+456")
        self.assertEqual(display.task_code, "123+456+123+456")

        # Reset task code
        display.set_task_code(None)
        self.assertIsNone(display.task_code)

        # Render frame with reset state (should render placeholder cards without crashing)
        img = display.render_frame()
        self.assertEqual(img.shape[:2], (480, 640))


if __name__ == "__main__":
    unittest.main()
