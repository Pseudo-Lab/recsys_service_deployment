from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from .models import ChatSession, ChatMessage
import json
import uuid

User = get_user_model()


class ChatSessionModelTest(TestCase):
    """ChatSession 모델 테스트"""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_create_session(self):
        """세션 생성 테스트"""
        session = ChatSession.objects.create(user=self.user)

        self.assertIsNotNone(session.id)
        self.assertIsInstance(session.id, uuid.UUID)
        self.assertEqual(session.title, 'New Chat')
        self.assertEqual(session.user, self.user)
        self.assertTrue(session.is_active)

    def test_create_session_without_user(self):
        """사용자 없이 세션 생성 (익명)"""
        session = ChatSession.objects.create()

        self.assertIsNotNone(session.id)
        self.assertIsNone(session.user)

    def test_session_soft_delete(self):
        """소프트 삭제 테스트"""
        session = ChatSession.objects.create(user=self.user)
        session.is_active = False
        session.save()

        # 세션은 DB에 여전히 존재
        self.assertTrue(ChatSession.objects.filter(id=session.id).exists())
        # 활성 세션 쿼리에서는 제외
        self.assertFalse(ChatSession.objects.filter(id=session.id, is_active=True).exists())


class ChatMessageModelTest(TestCase):
    """ChatMessage 모델 테스트"""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.session = ChatSession.objects.create(user=self.user)

    def test_create_message(self):
        """메시지 생성 테스트"""
        message = ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='한라산 맛집 추천해줘'
        )

        self.assertEqual(message.session, self.session)
        self.assertEqual(message.role, 'user')
        self.assertEqual(message.content, '한라산 맛집 추천해줘')
        self.assertEqual(message.metadata, {})

    def test_message_ordering(self):
        """메시지 순서 테스트"""
        msg1 = ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='첫 번째 메시지'
        )
        msg2 = ChatMessage.objects.create(
            session=self.session,
            role='assistant',
            content='두 번째 메시지'
        )

        messages = list(self.session.messages.all())
        self.assertEqual(messages[0], msg1)
        self.assertEqual(messages[1], msg2)

    def test_generate_title_from_first_message(self):
        """첫 메시지로 제목 생성 테스트"""
        ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='부모님과 성산일출봉 근처에서 3만원대 한정식 먹고 싶어요'
        )

        self.session.generate_title_from_first_message()

        # 50자 초과하면 ... 추가
        self.assertTrue(self.session.title.endswith('...'))
        self.assertLessEqual(len(self.session.title), 53)  # 50 + "..."

    def test_generate_title_short_message(self):
        """짧은 메시지로 제목 생성 테스트"""
        ChatMessage.objects.create(
            session=self.session,
            role='user',
            content='맛집 추천해줘'
        )

        self.session.generate_title_from_first_message()

        self.assertEqual(self.session.title, '맛집 추천해줘')


class SessionAPITest(TestCase):
    """세션 API 테스트"""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_session_list_unauthenticated(self):
        """비로그인 상태 세션 목록 조회"""
        response = self.client.get('/guiderec/api/sessions/')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['sessions'], [])
        self.assertFalse(data['authenticated'])

    def test_session_list_authenticated(self):
        """로그인 상태 세션 목록 조회"""
        self.client.login(username='testuser', password='testpass123')

        # 세션 2개 생성
        ChatSession.objects.create(user=self.user, title='첫 번째 대화')
        ChatSession.objects.create(user=self.user, title='두 번째 대화')

        response = self.client.get('/guiderec/api/sessions/')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data['sessions']), 2)
        self.assertTrue(data['authenticated'])

    def test_session_create_unauthenticated(self):
        """비로그인 상태 세션 생성"""
        response = self.client.post('/guiderec/api/sessions/create/')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn('session_id', data)
        self.assertFalse(data['authenticated'])

        # DB에는 저장되지 않음
        self.assertEqual(ChatSession.objects.count(), 0)

    def test_session_create_authenticated(self):
        """로그인 상태 세션 생성"""
        self.client.login(username='testuser', password='testpass123')

        response = self.client.post('/guiderec/api/sessions/create/')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn('session_id', data)
        self.assertTrue(data['authenticated'])

        # DB에 저장됨
        self.assertEqual(ChatSession.objects.count(), 1)

    def test_session_detail(self):
        """세션 상세 조회"""
        self.client.login(username='testuser', password='testpass123')

        session = ChatSession.objects.create(user=self.user, title='테스트 대화')
        ChatMessage.objects.create(
            session=session,
            role='user',
            content='테스트 메시지'
        )

        response = self.client.get(f'/guiderec/api/sessions/{session.id}/')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['title'], '테스트 대화')
        self.assertEqual(len(data['messages']), 1)

    def test_session_detail_not_found(self):
        """존재하지 않는 세션 조회"""
        self.client.login(username='testuser', password='testpass123')

        fake_id = uuid.uuid4()
        response = self.client.get(f'/guiderec/api/sessions/{fake_id}/')

        self.assertEqual(response.status_code, 404)

    def test_session_delete(self):
        """세션 삭제"""
        self.client.login(username='testuser', password='testpass123')

        session = ChatSession.objects.create(user=self.user)

        response = self.client.post(f'/guiderec/api/sessions/{session.id}/delete/')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

        # 소프트 삭제 확인
        session.refresh_from_db()
        self.assertFalse(session.is_active)

    def test_session_delete_unauthenticated(self):
        """비로그인 상태 세션 삭제 시도"""
        session = ChatSession.objects.create(user=self.user)

        response = self.client.post(f'/guiderec/api/sessions/{session.id}/delete/')

        self.assertEqual(response.status_code, 401)
