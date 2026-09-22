# Copyright © Michal Čihař <michal@weblate.org>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transaction boundaries for custom translation sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Barrier, Event
from time import monotonic, sleep
from typing import TYPE_CHECKING

from django.db import close_old_connections, connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from weblate.lang.models import Language
from weblate.trans.models import Unit, WorkflowSetting
from weblate.trans.models.source import (
    source_operation,
    source_project_gate,
    source_workflows,
)
from weblate.trans.tests.utils import RepoTestMixin, clear_users_cache
from weblate.utils.state import STATE_TRANSLATED

if TYPE_CHECKING:
    from collections.abc import Callable


def connected(operation: Callable[[], None]) -> None:
    close_old_connections()
    try:
        return operation()
    finally:
        connection.close()


class SourceOperationConcurrencyTest(RepoTestMixin, TransactionTestCase):
    def setUp(self) -> None:
        Language.objects.flush_object_cache()
        self.addCleanup(Language.objects.flush_object_cache)
        clear_users_cache()
        self.addCleanup(clear_users_cache)
        self.clone_test_repos()
        super().setUp()
        self.component = self.create_component()
        self.child = self.component.translation_set.get(
            language_code="cs"
        ).unit_set.order_by("pk")[0]
        self.parent = self.component.translation_set.get(
            language_code="de"
        ).unit_set.get(id_hash=self.child.id_hash)
        self.parent.target = "Old parent"
        self.parent.state = STATE_TRANSLATED
        self.parent.save()

    def wait_for_blocked(self, pid: int) -> None:
        deadline = monotonic() + 10
        while monotonic() < deadline:
            with connection.cursor() as cursor:
                cursor.execute("SELECT cardinality(pg_blocking_pids(%s)) > 0", [pid])
                if cursor.fetchone()[0]:
                    return
            sleep(0.01)
        self.fail("The competing operation did not wait for the project gate")

    def assert_source_consistent(self, *, configured: bool) -> None:
        child = Unit.objects.get(pk=self.child.pk)
        parent = Unit.objects.get(pk=self.parent.pk)
        self.assertEqual(child.translation_parent_id, parent.pk if configured else None)
        self.assertEqual(
            child.effective_source, parent.target if configured else child.source
        )
        self.assertFalse(child.translation_parent_blocked)
        if configured:
            self.assertEqual(
                child.details["translation_parent"]["applied"],
                child.source_snapshot.as_dict(),
            )

    def check_switch_race(self, action: str, *, edit_first: bool) -> None:
        workflow = None
        if action != "create":
            workflow = WorkflowSetting.objects.create(
                project=self.component.project,
                language=self.child.translation.language,
                source_language=(
                    self.parent.translation.language
                    if action == "delete"
                    else self.component.source_language
                ),
            )
        # Preserve a genuinely stale, separately loaded Project instance.
        self.assertEqual(
            bool(
                self.parent.translation.component.project.translation_parent_language_ids
            ),
            action != "create",
        )
        pid_queue: Queue[int] = Queue()
        edited = Event()
        release_edit = Event()

        def identify() -> None:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pid_queue.put(cursor.fetchone()[0])

        def edit() -> None:
            identify()
            with source_operation(self.parent.translation.component):
                self.parent.target = "New parent"
                self.parent.save()
                edited.set()
                if edit_first and not release_edit.wait(10):
                    msg = "Workflow did not attempt the exclusive gate"
                    raise TimeoutError(msg)

        def change_workflow() -> None:
            if action == "create":
                WorkflowSetting.objects.create(
                    project_id=self.component.project_id,
                    language_id=self.child.translation.language_id,
                    source_language_id=self.parent.translation.language_id,
                )
            elif action == "delete":
                assert workflow is not None
                WorkflowSetting.objects.filter(pk=workflow.pk).delete()
            else:
                assert workflow is not None
                current = WorkflowSetting.objects.get(pk=workflow.pk)
                current.source_language_id = self.parent.translation.language_id
                current.save()

        def configure() -> None:
            identify()
            change_workflow()

        with ThreadPoolExecutor(max_workers=2) as executor:
            if edit_first:
                edit_future = executor.submit(connected, edit)
                pid_queue.get(timeout=10)
                try:
                    self.assertTrue(edited.wait(10))
                    workflow_future = executor.submit(connected, configure)
                    self.wait_for_blocked(pid_queue.get(timeout=10))
                finally:
                    release_edit.set()
                edit_future.result(timeout=15)
                workflow_future.result(timeout=15)
            else:
                with source_project_gate([self.component.project_id], exclusive=True):
                    change_workflow()
                    edit_future = executor.submit(connected, edit)
                    self.wait_for_blocked(pid_queue.get(timeout=10))
                edit_future.result(timeout=15)
        self.assert_source_consistent(configured=action != "delete")

    def test_create_then_edit(self) -> None:
        self.check_switch_race("create", edit_first=False)

    def test_edit_then_create(self) -> None:
        self.check_switch_race("create", edit_first=True)

    def test_switch_then_edit(self) -> None:
        self.check_switch_race("switch", edit_first=False)

    def test_edit_then_switch(self) -> None:
        self.check_switch_race("switch", edit_first=True)

    def test_delete_then_edit(self) -> None:
        self.check_switch_race("delete", edit_first=False)

    def test_edit_then_delete(self) -> None:
        self.check_switch_race("delete", edit_first=True)

    def test_shared_gates_allow_concurrent_edits(self) -> None:
        ready = Barrier(2)

        def enter() -> None:
            with source_project_gate([self.component.project_id]):
                self.assertEqual(source_workflows(self.component.project_id), {})
                ready.wait(timeout=10)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(connected, enter)
            second = executor.submit(connected, enter)
            first.result(timeout=15)
            second.result(timeout=15)

    def test_rollback_restores_workflow_view(self) -> None:
        with source_project_gate([self.component.project_id], exclusive=True):
            with (
                self.assertRaisesMessage(ValueError, "rollback"),
                source_project_gate([self.component.project_id], exclusive=True),
            ):
                WorkflowSetting.objects.create(
                    project=self.component.project,
                    language=self.child.translation.language,
                    source_language=self.parent.translation.language,
                )
                self.assertTrue(source_workflows(self.component.project_id))
                msg = "rollback"
                raise ValueError(msg)
            self.assertEqual(source_workflows(self.component.project_id), {})
            self.assertFalse(
                WorkflowSetting.objects.filter(project=self.component.project).exists()
            )
        self.assert_source_consistent(configured=False)
        self.assertFalse(self.component.project.translation_parent_language_ids)

    def test_gate_migration_preserves_existing_data(self) -> None:
        workflow = WorkflowSetting.objects.create(
            project=self.component.project,
            language=self.child.translation.language,
            source_language=self.parent.translation.language,
        )
        before = list(Unit.objects.order_by("pk").values_list("pk", "state", "details"))
        try:
            MigrationExecutor(connection).migrate(
                [("trans", "0107_translation_parent")]
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT to_regprocedure('weblate_source_workflow_gate(bigint,boolean)')"
                )
                self.assertIsNone(cursor.fetchone()[0])
        finally:
            MigrationExecutor(connection).migrate(
                [("trans", "0108_source_workflow_gate")]
            )
        self.assertEqual(
            before,
            list(Unit.objects.order_by("pk").values_list("pk", "state", "details")),
        )
        with source_project_gate([self.component.project_id]):
            self.assertEqual(
                source_workflows(self.component.project_id),
                {workflow.language_id: workflow.source_language_id},
            )
